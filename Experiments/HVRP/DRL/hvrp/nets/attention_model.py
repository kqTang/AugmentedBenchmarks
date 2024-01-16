import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
from utils.tensor_functions import compute_in_batches

from nets.graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many
import copy
import random
import pdb
import numpy as np

def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
        )
        # return super(AttentionModelFixed, self).__getitem__(key)


class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 obj,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 instances='Choia&Tcha',
                 num_veh=6,
                 model_id=0):
        super(AttentionModel, self).__init__()
        self.instances = instances
        self.num_veh = num_veh if num_veh != None else 6
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.obj = obj
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.is_hcvrp = problem.NAME == 'hcvrp'
        self.feed_forward_hidden = 512

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        self.model_id = model_id
        # Problem specific context parameters (placeholder and step context dimension)
        # import pdb;pdb.set_trace()
        if self.is_hcvrp:
            # Embedding of last node + remaining_capacity / remaining length / remaining prize to collect
            step_context_dim = embedding_dim + 1
            # node_dim = 2 + self.num_veh  # x,y, demand(6 vehicles)
            # node_dim = 2 + self.num_veh  # x,y, demand(6 vehicles)
            # import pdb;pdb.set_trace()            
            if self.instances == 'Gloden':
                node_dim = 2 + self.num_veh*2  # x,y, demand(6 vehicles), cost
            elif (self.instances == 'Choia&Tcha') or (self.instances == 'Taillard'):
                node_dim = 2 + self.num_veh*3  # x,y, demand(6 vehicles), cost, a_cost  |  OR x,y, demand(6 vehicles), number, a_cost
            elif self.instances == 'DLP':
                node_dim = 2 + self.num_veh*3
            if (self.instances == 'Taillard') | (self.model_id==1):
                node_veh = 4 * self.num_veh
            elif self.model_id==0:
                node_veh = 3 * self.num_veh
            self.FF_veh = nn.Sequential(
                nn.Linear(node_veh, self.embedding_dim),
                nn.Linear(self.embedding_dim, self.feed_forward_hidden),
                nn.ReLU(),
                nn.Linear(self.feed_forward_hidden, self.embedding_dim)
            ) if self.feed_forward_hidden > 0 else nn.Linear(self.embedding_dim, self.embed_dim)

            self.FF_tour = nn.Sequential(
                nn.Linear(self.num_veh * self.embedding_dim, self.embedding_dim),
                nn.Linear(self.embedding_dim, self.feed_forward_hidden),
                nn.ReLU(),
                nn.Linear(self.feed_forward_hidden, self.embedding_dim)
            ) if self.feed_forward_hidden > 0 else nn.Linear(self.embedding_dim, self.embed_dim)
            self.select_embed = nn.Linear(self.embedding_dim * 2, self.num_veh)

            # Special embedding projection for depot node
            self.init_embed_depot = nn.Linear(2, embedding_dim)
            self.init_embed_ret = nn.Linear(2 * embedding_dim, embedding_dim)

        self.init_embed = nn.Linear(node_dim, embedding_dim)  # node_embedding

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(
            embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(
            embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(
            step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """
        # embeddings: [batch_size, graph_size+1, embed_dim]

        if self.checkpoint_encoder:
            embeddings, _ = checkpoint(self.embedder, self._init_embed(
                input))  # self._init_embed(input): [batch_size, graph_size+1, embed_dim]
        else:
            embeddings, _ = self.embedder(self._init_embed(input))
        # 
        # import pdb;pdb.set_trace()
        _log_p, log_p_veh, pi, veh_list, tours = self._inner(
            input, embeddings)  # _log_p: [batch_size, graph_size+1, graph_size+1], pi:[batch_size, graph_size+1]
        # pdb.set_trace()
        cost, mask = self.problem.get_costs(input, self.obj, pi, veh_list, tours, self.instances, self.num_veh)  # mask is None, cost:[batch_size]

        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll, ll_veh = self._calc_log_likelihood(
            _log_p, log_p_veh, pi, mask, veh_list)  # [batch_size], 所有被选点对应的log_pro的和
        if return_pi:
            return cost, ll, ll_veh, pi

        return cost, ll, ll_veh

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input):
        embeddings, _ = self.embedder(self._init_embed(input))
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        # First dim = batch_size * cur_beam_size
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(
                fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        # This will broadcast, calculate log_p (score) of expansions
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        # We flatten the action as we need to filter and this cannot be done in 2d
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10  # != -math.inf triggers

        # Parent is row idx of ind_topk, can be found by enumerating elements and dividing by number of columns
        flat_parent = torch.arange(
            flat_action.size(-1), out=flat_action.new()) / ind_topk.size(-1)

        # Filter infeasible
        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            # Too bad, no feasible expansions at all :(
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

    def _calc_log_likelihood(self, _log_p, _log_p_veh, a, mask, veh_list):  # a is pi

        log_p = _log_p.gather(2, torch.tensor(a).unsqueeze(-1)).squeeze(-1)
        log_p_veh = _log_p_veh.gather(2, torch.tensor(
            veh_list).cuda().unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0
            log_p_veh[mask] = 0
        
        if not (log_p_veh >= -1000).data.all():
            pdb.set_trace()
        assert (
            log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"
        assert (
            log_p_veh > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1), log_p_veh.sum(1)  # [batch_size]

    def _init_embed(self, input):

        if self.is_hcvrp:
            # pdb.set_trace()
            # 他这里默认所有案例的车辆都一致，但是我们不是的，因此，对于每个案例，都需要用对应的需求来除以对应的车辆负重
            # demand = torch.tensor([(input['demand'] / input['capacity'][0, veh]).tolist() for veh in range(input['capacity'].size(-1))]).transpose(0, 1).transpose(1, 2).cuda()
            demand = torch.tensor(input['demand'].unsqueeze(2) / (input['capacity']+1e-5).unsqueeze(1))    
            if self.instances != 'Taillard':  
                fixed_cost =  input['fixed_cost'].unsqueeze(1).expand_as(demand)
            if self.instances == 'Choia&Tcha':
                a_cost =  input['a_cost'].unsqueeze(1).expand_as(demand)
                return torch.cat(  # [batch_size, graph_size+1, embed_dim]
                (
                    self.init_embed_depot(input['depot'])[:, None, :],
                    self.init_embed(torch.cat((  # [batch_size, graph_size, embed_dim]
                        input['loc'],  # [batch_size, graph_size, 2]        2
                        demand,  # [batch_size, graph_size, num_veh]        6
                        fixed_cost, # [batch_size, graph_size, num_veh]     6
                        a_cost, # [batch_size, graph_size, num_veh]         6
                    ), -1))
                ),
                1
            )
            elif self.instances == 'Taillard':
                a_cost =  input['a_cost'].unsqueeze(1).expand_as(demand)
                number =  input['number'].unsqueeze(1).expand_as(demand)
                return torch.cat(  # [batch_size, graph_size+1, embed_dim]
                (
                    self.init_embed_depot(input['depot'])[:, None, :],
                    self.init_embed(torch.cat((  # [batch_size, graph_size, embed_dim]
                        input['loc'],  # [batch_size, graph_size, 2]        2
                        demand,  # [batch_size, graph_size, num_veh]        6
                        number, # [batch_size, graph_size, num_veh]     6
                        a_cost, # [batch_size, graph_size, num_veh]         6
                    ), -1))
                ),
                1
            )                
            # import pdb
            # pdb.set_trace()

            return torch.cat(  # [batch_size, graph_size+1, embed_dim]
                (
                    self.init_embed_depot(input['depot'])[:, None, :],
                    self.init_embed(torch.cat((  # [batch_size, graph_size, embed_dim]
                        input['loc'],  # [batch_size, graph_size, 2]
                        demand,  # [batch_size, graph_size, num_veh]
                        fixed_cost, # [batch_size, graph_size, num_veh]
                    ), -1))
                ),
                1
            )

    def select_veh(self, input, state, sequences, embeddings, obj, veh_list, tours):
        current_node = state.get_current_node()  # [batch_size]
        tour_dis = state.lengths  # [batch_size, num_veh]
        # if obj == 'min-max':
        #     SPEED = [1, 1, 1, 1, 1, 1]
        # elif obj == 'min-sum':
            # SPEED = [1/4, 1/5, 1/6, 1/7, 1/8]
        if self.instances=='Gloden':
            SPEED = torch.ones_like(tour_dis)
            # [1, 1, 1, 1, 1, 1]
        else:
            SPEED = input['a_cost']

        # pdb.set_trace()
        batch_size, _, embed_dim = embeddings.size()
        _, num_veh = current_node.size()
   
        if sequences:
            # [batch_size, tour_len]
            # pdb.set_trace()
            tours = [torch.stack(tour, -1).squeeze(-2) for tour in tours]
            tour_con = [torch.gather(embeddings,1,(tour.clone())[..., None].contiguous().expand(batch_size, tour.size(-1), embed_dim)).view(batch_size, tour.size(-1), embed_dim) for tour in tours]  # [batch_size, tour_len, embed_dim]
            
            mean_tour = torch.cat(  # [batch_size, 3*embed_dim]
                (
                    [torch.max(tour_con_i, dim=1)[0] for tour_con_i in tour_con]
                ),
                1,
            )  # [batch_size, embed_dim]

            current_loc = state.coords.gather(  # [batch_size, graph_size, 2]
                1,
                (current_node.clone())[..., None].contiguous()
                .expand_as(state.coords[:, 0:num_veh, :])
            ).transpose(0, 1)  # [num_veh, batch_size, 2]
            

            # import pdb;pdb.set_trace()
            
            tour = torch.cat(([tour.unsqueeze(1) for tour in tours]),dim=1)
            tour_d = tour[...,:-1] - tour[...,1:]
            if self.instances!='Taillard':
                '''
                011203300000
                100010011111
                11203300000
                01120330000
                11111110000
                
                00444999500
                000887000
                '''                
                fixed_cost=(((tour[...,:-1] == 0) & (tour_d != 0)).to(dtype=torch.bool).sum(-1) * input['fixed_cost'])     
                # print('-----------fixed_cost',fixed_cost[0],'fixed_cost.shape',fixed_cost.shape,tour_d[0]) #第一次选择一个车辆似乎不会增加fixed_cost？
            else:
                number_limit = input['number']  # Number limit of each vehicle type
                used_number = ((tour[..., 1:] == 0) & (tour_d != 0)).to(dtype=torch.bool).sum(-1)  # Used number of each vehicle type
                remained_number = number_limit - used_number   # Compute exceed number     
     
                            
            # if (tours[0].size(1) != 1):
            #     # import pdb
            #     # pdb.set_trace()
            #     tour = torch.cat(([tour.unsqueeze(1) for tour in tours]),dim=1)
            #     tour_d = tour[...,:-1] - tour[...,1:]
            #     if self.instances!='Taillard':
            #         fixed_cost=(((tour[...,1:] == 0) & (tour_d != 0)).to(dtype=torch.bool).sum(-1) * input['fixed_cost'])     
            #     else:
            #         number_limit = input['number']  # Number limit of each vehicle type
            #         used_number = ((tour[..., 1:] == 0) & (tour_d != 0)).to(dtype=torch.bool).sum(-1)  # Used number of each vehicle type
            #         remained_number = number_limit - used_number   # Compute exceed number
            # else:
            #     remained_number = input['number'] if self.instances=='Taillard' else None
            #     fixed_cost=(((tour[...,1:] == 0) & (tour_d != 0)).to(dtype=torch.bool).sum(-1) * input['fixed_cost'])     
            
            if self.instances!='Taillard':
                if self.model_id==0:
                    veh_context = torch.cat(  # [batch_size, num_veh, 5]
                    (
                        #veh_index[None, :].expand(batch_size, veh_index.size(-1))[:, 0].unsqueeze(-1),
                        # [batch_size, num_veh]
                       [torch.cat([tour_dis[:, i].unsqueeze(-1) * SPEED[:,i:i+1]+fixed_cost[:,i:i+1],current_loc[i, :]],dim=-1)
                        for i in range(len(tours))]  # [batch_size, 2])                 
                    ),
                    -1
                )
                else:
                    veh_context = torch.cat(  # [batch_size, num_veh, 5]
                        (
                            #veh_index[None, :].expand(batch_size, veh_index.size(-1))[:, 0].unsqueeze(-1),
                            # [batch_size, num_veh]
                            [torch.cat([tour_dis[:, i].unsqueeze(-1) * SPEED[:,i:i+1],fixed_cost[:,i:i+1],current_loc[i, :]],dim=-1)
                            for i in range(len(tours))]  # [batch_size, 2])                 
                        ),
                        -1
                    )
            elif self.instances=='Taillard': #Taillard, has 'number', does not have 'fixed_cost'
                veh_context = torch.cat(  # [batch_size, num_veh, 5]
                    (
                        #veh_index[None, :].expand(batch_size, veh_index.size(-1))[:, 0].unsqueeze(-1),
                        # [batch_size, num_veh]
                        [torch.cat([tour_dis[:, i].unsqueeze(-1) * SPEED[:,i:i+1],
                                    remained_number[:,i:i+1],
                                    current_loc[i, :]],dim=-1)
                        for i in range(len(tours))]  # [batch_size, 2])                 
                    ),
                    -1
                )
            # print("----------",veh_context.shape)
            # pdb.set_trace()
        else:
            current_loc = state.coords.gather(  # [batch_size, graph_size, 2]
                1,
                (current_node.clone())[..., None].contiguous()
                .expand_as(state.coords[:, 0:self.num_veh, :])
            ).transpose(0, 1)  # [batch_size, 2]
            mean_tour = torch.zeros([batch_size, self.num_veh * embed_dim]).float().cuda()                    
            if self.instances=='Taillard':
                # import pdb;pdb.set_trace()
                veh_context = torch.cat(  # [batch_size, num_veh, 6]
                (
                    # [batch_size, num_veh]
                    [torch.cat([tour_dis[:, i].unsqueeze(-1) * SPEED[:,i:i+1],
                                input['number'][:,i:i+1],
                                current_loc[i, :]],dim=-1)
                     for i in range(len(tours))]
                ),
                
                -1
            )
            else :
                if self.model_id==0:
                    veh_context = torch.cat(  # [batch_size, num_veh, 5]
                    (
                        #veh_index[None, :].expand(batch_size, veh_index.size(-1))[:, 0].unsqueeze(-1),
                        # [batch_size, num_veh]
                        [torch.cat([tour_dis[:, i].unsqueeze(-1) * SPEED[:,i:i+1],current_loc[i, :]],dim=-1)
                        for i in range(len(tours))]  # [batch_size, 2])                 
                    ),
                    -1
                )
                else:
                    veh_context = torch.cat(  # [batch_size, num_veh, 6]
                    (
                        # [batch_size, num_veh]
                        [torch.cat([tour_dis[:, i].unsqueeze(-1) * SPEED[:,i:i+1],
                                    torch.zeros_like(tour_dis[:, i].unsqueeze(-1)),
                                    current_loc[i, :]],dim=-1)
                        for i in range(len(tours))]
                    ),
                    
                    -1
                )
        # print("----------",veh_context.shape)
        # import pdb;pdb.set_trace()
        # print(veh_context.shape)
        veh_context = self.FF_veh(veh_context)
        tour_context = self.FF_tour(mean_tour)
        context = torch.cat((veh_context, tour_context), -
                            1).view(batch_size, self.embedding_dim * 2)
        # mask_veh = (self.capacity[torch.arange(batch_size), veh].unsqueeze(-1)[..., None].expand_as(self.demand[self.ids, 1:]) == 0)
        unvisited_demand=torch.where(
            state.visited.squeeze(1),
            torch.ones_like(state.demand) * 1000.,
            state.demand
        )
        # import pdb;pdb.set_trace() 
        if  (self.instances=='Taillard'):
            # import pdb;pdb.set_trace()
            mask_veh = (state.capacity==0) | ((state.capacity<unvisited_demand.min(1)[0].unsqueeze(1)) & (~(state.visited[...,1:].all(2).expand_as(state.capacity)))).to(dtype=torch.bool) | (remained_number<=0)
            if (remained_number<0).any():
                import pdb;pdb.set_trace()
        else:
            mask_veh = (state.capacity==0) | ((state.capacity<unvisited_demand.min(1)[0].unsqueeze(1)) & (~(state.visited[...,1:].all(2).expand_as(state.capacity)))).to(dtype=torch.bool)
        logits=self.select_embed(context)
        
        logits[mask_veh] += -np.inf
        
        log_veh = F.log_softmax(logits, dim=1)
        p_veh = F.softmax(logits, dim=1)
        if not p_veh.min() >=0:
            import pdb;pdb.set_trace() 
        

        if self.decode_type == "greedy":
            veh = torch.max(p_veh, dim=1)[1]
        elif self.decode_type == "sampling":
            veh = p_veh.multinomial(1).squeeze(-1)
        # log_veh[mask_veh] = 0
        # p_veh[mask_veh] = 0
        # pdb.set_trace()        
        return veh, log_veh

    def _inner(self, input, embeddings):
        # input: [batch_size, graph_size, node_dim], node_dim=2, location
        # embeddings: [batch_size, graph_size+1, embed_dim]
        state = self.problem.make_state(input)
        current_node = state.get_current_node()
        batch_size, num_veh = current_node.size()

        outputs = []
        outputs_veh = []
        sequences = []
        tours = [[] for i in range(num_veh)]
        for i in range(num_veh): tours[i].append(torch.zeros((batch_size),device=current_node.device,dtype=torch.long)) 

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(
            embeddings)  # embeddings, context_node_project(graph_embed), glimpse_key, glimpse_val, logits_key

        # Perform decoding steps
        i = 0
        #veh = torch.LongTensor(batch_size).zero_()
        veh_list = []
        while not (self.shrink_size is None and state.all_finished()):
            # pdb.set_trace()
            veh, log_p_veh = self.select_veh(input, state, sequences, embeddings, self.obj,
                                             veh_list, tours)  # [batch_size, 1]
            #veh = torch.min(state.lengths, dim=-1)[1]
            veh_list.append(veh.tolist())
            if self.shrink_size is not None:
                unfinished = torch.nonzero(state.get_finished() == 0)
                if len(unfinished) == 0:
                    break
                unfinished = unfinished[:, 0]
                # Check if we can shrink by at least shrink_size and if this leaves at least 16
                # (otherwise batch norm will not work well and it is inefficient anyway)
                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                    # Filter states
                    state = state[unfinished]
                    fixed = fixed[unfinished]
            # log_p: [batch_size, num_step, graph_size], mask:[batch_size, num_step, graph_size]
            log_p, mask = self._get_log_p(fixed, state, veh)

            # Select the indices of the next nodes in the sequences, result (batch_size) long
            selected = self._select_node(log_p.exp(
            )[:, 0, :], mask[:, 0, :], state, veh, sequences)  # Squeeze out steps dimension

            state = state.update(selected, veh)

            # Now make log_p, selected desired output size by 'unshrinking'
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)

                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_
            # Collect output of step
            outputs.append(log_p[:, 0, :])
            outputs_veh.append(log_p_veh)

            sequences.append(selected[torch.arange(batch_size), veh])
            # tours = [tours[i].append(selected[:, i]) for i in range(num_veh)]
            for i in range(num_veh): tours[i].append(selected[:, i]) 

            i += 1
        veh_list = torch.tensor(veh_list).transpose(0, 1).cuda()
        # output:[batch_size, solu_len, graph_size+1], sequences: [batch_size, tour_len]
        return torch.stack(outputs, 1), torch.stack(outputs_veh, 1), torch.stack(sequences, -1).squeeze(-2), veh_list, [torch.stack(tours[i], -1) for i in range(num_veh)]

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        # print('input', input)
        
        return sample_many(
            # Need to unpack tuple into arguments
            lambda input: self._inner(*input),
            lambda input, pi, veh_list, tours: self.problem.get_costs(
                input[0], self.obj, pi, veh_list, tours, self.instances, self.num_veh),  # Don't need embeddings as input to get_costs
            # Pack input with embeddings (additional input)
            (input, self.embedder(self._init_embed(input))[0]),
            batch_rep, iter_rep
        )

    # probs, mask: [batch_size, graph_size]
    def _select_node(self, probs, mask, state, veh, sequences):
        assert (probs == probs).all(), "Probs should not contain any nans"

        selected = (state.get_current_node()).clone()
        batch_size, _ = (state.get_current_node()).size()

        if self.decode_type == "greedy":
            _, selected[torch.arange(batch_size), veh] = probs.max(1)
            assert not mask.gather(-1, selected[torch.arange(batch_size), veh].unsqueeze(-1)).data.any(
            ), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected[torch.arange(batch_size), veh] = probs.multinomial(1).squeeze(
                1)  # [batch_size]

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(-1, selected[torch.arange(batch_size), veh].unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected[torch.arange(batch_size), veh] = probs.multinomial(
                    1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def _precompute(self, embeddings, num_steps=1):
        # embeddings: [batch_size, graph_size+1, embed_dim]

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)  # [batch_size, embed_dim]
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(
            graph_embed)[:, None, :]  # linear(graph_embed)

        # The projection of the node embeddings for the attention is calculated once up front
        # glimpse_key_fixed size is torch.Size([batch_size, 1, graph_size+1, embed_dim])
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3,
                                                                          dim=-1)  # split tensor to three parts in dimension 1

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (  # make multihead
            # (n_heads, batch_size, num_steps, graph_size, embed_dim)
            self._make_heads(glimpse_key_fixed, num_steps),
            # (n_heads, batch_size, num_steps, graph_size, embed_dim)
            self._make_heads(glimpse_val_fixed, num_steps),
            # [batch_size, 1, graph_size+1, embed_dim]
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        # Return all, note different from torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device,
                         dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )

    def _get_log_p(self, fixed, state, veh, normalize=True):
        # fixed: node_embeddings(embeddings), context_node_project(graph_embed), glimpse_key, glimpse_val, logits_key
        # Compute query = context node embedding, 相同维度数字相加
        # fixed.context_node_projected (graph_embedding): (batch_size, 1, embed_dim), query: [batch_size, num_veh, embed_dim]
        query = fixed.context_node_projected + \
            self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings,
                                                                      state, veh))  # after project: [batch_size, 1, embed_dim]
        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(
            fixed, state)

        # Compute the mask
        mask = state.get_mask(veh)  # [batch_size, 1, graph_size]

        # Compute logits (unnormalized log_p)  log_p:[batch_size, num_veh, graph_size], glimpse:[batch_size, num_veh, embed_dim]
        log_p, glimpse = self._one_to_many_logits(
            query, glimpse_K, glimpse_V, logit_K, mask, veh)

        if normalize:
            log_p = F.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state, veh, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)

        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        current_node = (state.get_current_node()).clone()
        batch_size, num_veh = current_node.size()
        num_steps = 1

        if self.is_hcvrp:
            # import pdb
            # pdb.set_trace()            
            # Embedding of previous node + remaining capacity
            if from_depot:
                # 1st dimension is node idx, but we do not squeeze it since we want to insert step dimension
                # i.e. we actually want embeddings[:, 0, :][:, None, :] which is equivalent

                return torch.cat(  # [batch_size, num_veh, embed_dim+1]
                    (
                        embeddings[:, 0:1, :].expand(
                            batch_size, num_veh, embeddings.size(-1)),
                        # used capacity is 0 after visiting depot
                        state.capacity.unsqueeze(-1) - torch.zeros_like(state.used_capacity[:, :, None])
                    ),
                    -1
                )
            else:
                return torch.cat(  # [batch_size, num_veh, embed_dim+1]
                    (
                        torch.gather(
                            embeddings,  # [batch_size, graph_size, embed_dim]
                            1,
                            
                            (current_node[torch.arange(
                                batch_size), veh]).contiguous()
                            .view(batch_size, num_steps, 1)
                            .expand(batch_size, num_steps, embeddings.size(-1))
                        ).view(batch_size, num_steps, embeddings.size(-1)),  # [batch_size, num_step, embed_dim]
                        
                        # (state.capacity[torch.arange(batch_size),veh] -  state.used_capacity[torch.arange(batch_size), veh]).transpose(0, 1).unsqueeze(-1)
                        (state.capacity[torch.arange(batch_size),veh] -  state.used_capacity[torch.arange(batch_size), veh])[:,None,None]
                    ),
                    -1
                )

        if self.is_pdvrp:
            # Embedding of previous node + remaining capacity
            if from_depot:
                # 1st dimension is node idx, but we do not squeeze it since we want to insert step dimension
                # i.e. we actually want embeddings[:, 0, :][:, None, :] which is equivalent
                return torch.cat(  # [batch_size, num_steps, 2*embed_dim] step_contex_dim
                    (
                        embeddings[:, 0:1, :].expand(
                            batch_size, num_steps, embeddings.size(-1)),
                        torch.gather(
                            embeddings,
                            1,
                            current_node.contiguous().view(batch_size, num_steps, 1)
                            .expand(batch_size, num_steps, embeddings.size(-1))
                            # [batch_size, num_steps, embed_dim]
                        ).view(batch_size, num_steps, embeddings.size(-1))
                    ),
                    -1
                )
            else:
                return torch.gather(
                    embeddings,
                    1,
                    current_node.contiguous()
                    .view(batch_size, num_steps, 1)
                    .expand(batch_size, num_steps, embeddings.size(-1))
                ).view(batch_size, num_steps, embeddings.size(-1))

        if self.is_vrp:
            # Embedding of previous node + remaining capacity
            if from_depot:
                # 1st dimension is node idx, but we do not squeeze it since we want to insert step dimension
                # i.e. we actually want embeddings[:, 0, :][:, None, :] which is equivalent
                return torch.cat(
                    (
                        embeddings[:, 0:1, :].expand(
                            batch_size, num_steps, embeddings.size(-1)),
                        # used capacity is 0 after visiting depot
                        self.problem.VEHICLE_CAPACITY - \
                        torch.zeros_like(state.used_capacity[:, :, None])
                    ),
                    -1
                )
            else:
                return torch.cat(
                    (
                        torch.gather(
                            embeddings,  # [batch_size, graph_size, embed_dim]
                            1,
                            current_node.contiguous()
                            .view(batch_size, num_steps, 1)
                            .expand(batch_size, num_steps, embeddings.size(-1))
                        ).view(batch_size, num_steps, embeddings.size(-1)),  # [batch_size, num_steps, embed_dim]
                        self.problem.VEHICLE_CAPACITY - \
                        state.used_capacity[:, :, None]
                    ),
                    -1
                )

        elif self.is_orienteering or self.is_pctsp:
            return torch.cat(
                (
                    torch.gather(
                        embeddings,
                        1,
                        current_node.contiguous()
                        .view(batch_size, num_steps, 1)
                        .expand(batch_size, num_steps, embeddings.size(-1))
                    ).view(batch_size, num_steps, embeddings.size(-1)),
                    (
                        state.get_remaining_length()[:, :, None]
                        if self.is_orienteering
                        else state.get_remaining_prize_to_collect()[:, :, None]
                    )
                ),
                -1
            )

        else:  # TSP

            if num_steps == 1:  # We need to special case if we have only 1 step, may be the first or not
                if state.i.item() == 0:
                    # First and only step, ignore prev_a (this is a placeholder)
                    return self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
                else:
                    return embeddings.gather(
                        1,
                        torch.cat((state.first_a, current_node), 1)[:, :, None].expand(batch_size, 2,
                                                                                       embeddings.size(-1))
                    ).view(batch_size, 1, -1)
            # More than one step, assume always starting with first
            embeddings_per_step = embeddings.gather(
                1,
                current_node[:, 1:, None].expand(
                    batch_size, num_steps - 1, embeddings.size(-1))
            )
            return torch.cat((
                # First step placeholder, cat in dim 1 (time steps)
                self.W_placeholder[None, None, :].expand(
                    batch_size, 1, self.W_placeholder.size(-1)),
                # Second step, concatenate embedding of first with embedding of current/previous (in dim 2, context dim)
                torch.cat((
                    embeddings_per_step[:, 0:1, :].expand(
                        batch_size, num_steps - 1, embeddings.size(-1)),
                    embeddings_per_step
                ), 2)
            ), 1)

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask, veh):
        batch_size, num_step, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads  # query and K both have key_size

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_step, 1, key_size)
        glimpse_Q = query.view(
            batch_size, num_step, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_step, 1, graph_size)
        # glimpse_K (n_heads, batch_size, 1, graph_size, key_size)
        compatibility = torch.matmul(
            glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))

        if self.mask_inner:  # True
            assert self.mask_logits, "Cannot mask inner without masking logits"  # True
            # mask: # [batch_size, num_veh, graph_size]
            # mask visited nodes and nodes cannot be visited
            compatibility[mask[None, :, :, None, :].expand_as(
                compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_step, 1, val_size)
        heads = torch.matmul(F.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_step, 1, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_step, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        final_Q = glimpse
        # logits_K, (batch_size, 1, graph_size, embed_dim)
        # Batch matrix multiplication to compute logits (batch_size, num_step, graph_size)
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)
                              ).squeeze(-2) / math.sqrt(final_Q.size(-1))
        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:  # 10
            # print*(F.tanh(logits))
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:  # True
            logits[mask] = -math.inf

        # glimpse[batch_size, num_veh, embed_dim]
        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed, state):

        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    # v: [batch_size, 1, graph_size+1, embed_dim]
    def _make_heads(self, v, num_steps=1):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            # (n_heads, batch_size, num_steps, graph_size, embed_dim)
            .permute(3, 0, 1, 2, 4)
        )
