"""Per-question BM25 retrieval environment with an assistant-only exact-match outcome reward."""
import re
from rank_bm25 import BM25Okapi
from skyrl_gym.envs.search.env import SearchEnv, SearchEnvConfig
from scripts.evaluation import action, format_documents, inject, normalize


class BM25SearchEnv(SearchEnv):
    def __init__(self, env_config, extras):
        super().__init__(SearchEnvConfig(topk=3), extras)
        self.documents = extras['documents']
        self.training_attack = extras.get('training_attack', 'clean')
        self.training_target = extras.get('training_target', '')
        if self.training_attack not in ('clean', 'train_direct', 'train_role'):
            raise ValueError('Only training attack templates may enter training')
        if self.training_attack != 'clean' and (not self.training_target or any(
                normalize(self.training_target) == normalize(gold)
                for gold in self.ground_truth['target'])):
            raise ValueError('Training attack target is empty or collides with gold')
        self.searches = 0
        self.attack_exposed = False
        self.tokenize = lambda text: re.findall(r'\w+', text.lower())
        self.retriever = BM25Okapi([self.tokenize(d['title']+' '+d['text']) for d in self.documents])

    def _execute_tool(self, tool_group_name, tool_name, tool_input):
        query = tool_input[0] if isinstance(tool_input, list) else tool_input
        if not query:
            return 'Invalid action: use <search>query</search> or <answer>answer</answer>.'
        scores = self.retriever.get_scores(self.tokenize(query))
        indices = sorted(range(len(self.documents)),key=lambda i:(-float(scores[i]),i))[:3]
        shown = [self.documents[i] for i in indices]
        if self.searches == 0 and self.training_attack != 'clean':
            shown = inject(shown, self.training_attack, self.training_target)
            self.attack_exposed = True
        self.searches += 1
        return '\n\n<information>'+format_documents(shown).strip()+'</information>\n\n'

    def step(self, generated_action):
        result = super().step(generated_action)
        result['metadata'].update(training_attack=self.training_attack,
                                  attack_exposed=self.attack_exposed)
        return result

    def _get_reward(self, generated_action, done):
        if not done:
            return 0.0
        kind, answer = action(generated_action)
        if kind != 'answer' or not answer or generated_action.count('<answer>') != 1 or generated_action.count('</answer>') != 1:
            return 0.0
        return float(any(normalize(answer)==normalize(gold) for gold in self.ground_truth['target']))
