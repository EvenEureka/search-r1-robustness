"""Batch-scoped first-search sharing; independent continuation requests stay on policy."""
import asyncio
import copy
import hashlib
import json
from dataclasses import dataclass, field
from scripts.evaluation import action


def token_digest(ids):
    return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()


def selected_group(question_id, step, seed, fraction):
    assert 0 <= fraction <= 1
    h=hashlib.sha256(f'{seed}:{step}:{question_id}'.encode()).digest()
    return int.from_bytes(h[:8],'big')/2**64 < fraction


@dataclass
class BoundaryGroup:
    uid: str
    step: int
    future: object = field(default_factory=lambda:asyncio.get_running_loop().create_future())
    shared: bool = False
    prompt_ids: list = field(default_factory=list)
    first_ids: list = field(default_factory=list)
    query: str = ''
    clean_documents: list = field(default_factory=list)
    template: str = 'train_direct'


@dataclass
class BranchSession:
    group: BoundaryGroup
    repetition: int
    calls: int = 0
    continuation_input: list = field(default_factory=list)
    backend_generated_tokens: int = 0


class BoundaryClient:
    """Proxy sharing only the selected group's first actual search generation."""
    def __init__(self,backend,sessions):
        self.backend=backend
        self.sessions=sessions

    def __getattr__(self,name):
        return getattr(self.backend,name)

    async def _generate(self,request,model,session):
        out=await self.backend.generate(request,model=model)
        if session is not None:
            session.backend_generated_tokens+=sum(len(x) for x in out['response_ids'])
        return out

    async def generate(self,request,model=None):
        session=self.sessions.get(request['session_ids'][0])
        if session is None:return await self._generate(request,model,None)
        session.calls+=1
        group=session.group
        if session.calls>1:
            if session.calls==2 and group.shared:
                session.continuation_input=list(request['prompt_token_ids'][0])
            return await self._generate(request,model,session)
        if session.repetition==0:
            try:
                output=await self._generate(request,model,session)
                kind,query=action(output['responses'][0])
                # Parent SearchEnv stops on a complete answer, so disallow ambiguous dual actions.
                group.shared=(kind=='search' and bool(query) and '</answer>' not in output['responses'][0]
                              and output['stop_reasons'][0]=='stop')
                group.prompt_ids=list(request['prompt_token_ids'][0])
                group.first_ids=list(output['response_ids'][0])
                group.query=query if group.shared else ''
                group.future.set_result(copy.deepcopy(output))
                return output
            except BaseException as error:
                if not group.future.done():group.future.set_exception(error)
                raise
        shared_output=await group.future
        assert request['prompt_token_ids'][0]==group.prompt_ids,'Different prompts in a paired group'
        if group.shared:return copy.deepcopy(shared_output)
        # No search in the anchor: remaining rollouts get their own first generation.
        return await self._generate(request,model,session)


def mask_shared_prefix(response_ids,loss_mask,first_ids,eos_token_id):
    ids=first_ids[:-1] if first_ids and first_ids[-1]==eos_token_id else first_ids
    assert response_ids[:len(ids)]==ids,'Shared prefix changed during token assembly'
    result=list(loss_mask)
    result[:len(ids)]=[0]*len(ids)
    return result,len(ids)
