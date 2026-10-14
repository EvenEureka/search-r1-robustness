"""Training-only branching metadata never appears in the model observation."""
import copy
from training_adapter.search_env import BM25SearchEnv


class BoundarySearchEnv(BM25SearchEnv):
    def __init__(self,env_config,extras):
        super().__init__(env_config,extras)
        self.boundary_session=extras.get('_boundary_session')

    def _execute_tool(self,tool_group_name,tool_name,tool_input):
        session=self.boundary_session
        if session is not None:
            group=session.group
            self.training_attack=group.template if group.shared and session.repetition>=2 else 'clean'
            if self.searches==0 and group.shared:
                query=tool_input[0] if isinstance(tool_input,list) else tool_input
                assert query.strip()==group.query.strip()
                scores=self.retriever.get_scores(self.tokenize(query))
                indices=sorted(range(len(self.documents)),key=lambda i:(-float(scores[i]),i))[:3]
                docs=[self.documents[i] for i in indices]
                if not group.clean_documents:group.clean_documents=copy.deepcopy(docs)
                assert docs==group.clean_documents,'Paired retrieval evidence differs'
        return super()._execute_tool(tool_group_name,tool_name,tool_input)
