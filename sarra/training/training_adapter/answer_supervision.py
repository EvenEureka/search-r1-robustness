"""Training-only answer targets with explicit provenance and abstention rules."""
from collections import Counter
from scripts.evaluation import action,normalize


def last_generated_text(tokens,mask,tokenizer):
    indices=[i for i,m in enumerate(mask) if m]
    if not indices:return ''
    end=indices[-1]+1;start=end-1
    while start>0 and mask[start-1]:start-=1
    return tokenizer.decode(tokens[start:end],skip_special_tokens=True)


def select_answer_examples(records,output,tokenizer,support_labels,max_length):
    groups={};counts=Counter();examples=[]
    for row in records:
        if row['shared']:groups.setdefault(row['group'],[]).append(row)
    for uid,rr in groups.items():
        rr.sort(key=lambda r:r['repetition'])
        assert [r['repetition'] for r in rr]==[0,1,2,3],'Minibatch split a prompt group'
        counts['groups_considered']+=1
        answer=None;source=None
        for clean in rr[:2]:
            i=clean['row']
            text=last_generated_text(output['response_ids'][i],output['loss_masks'][i],tokenizer)
            kind,candidate=action(text)
            if clean['reward']>.5 and kind=='answer' and candidate and any(normalize(candidate)==normalize(g) for g in clean['ground_truth']):
                answer=candidate;source='gold_checked_clean_answer';break
        if answer is None:
            required=set(support_labels.get(rr[0].get('question_id'),[]))
            present={d['title'] for d in rr[0]['clean_documents']}
            if required and required<=present:
                answer=rr[0]['ground_truth'][0];source='training_gold_with_support_title_coverage'
            else:
                counts['skipped_no_verified_target']+=1;continue
        # The attacked boundary is identical for reps 2 and 3: supervise it once.
        ids=rr[2]['continuation_input_ids']
        if not ids:counts['skipped_no_continuation']+=1;continue
        prompt=ids+tokenizer.encode('\n<answer>',add_special_tokens=False)
        target=tokenizer.encode(' '+answer,add_special_tokens=False)
        suffix=tokenizer.encode('</answer>',add_special_tokens=False)+[tokenizer.eos_token_id]
        if len(prompt)+len(target)+len(suffix)>max_length:
            counts['skipped_context_limit']+=1;continue
        assert target
        examples.append({'group':uid,'question_id':rr[0].get('question_id'),'source':source,'answer':answer,
            'prompt_ids':prompt,'response_ids':target+suffix,'answer_mask':[1]*len(target)+[0]*len(suffix)})
        counts[source]+=1
    counts['examples']=len(examples)
    counts['answer_tokens']=sum(sum(x['answer_mask']) for x in examples)
    return examples,dict(counts)
