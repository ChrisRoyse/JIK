"""Input-only lexical and persistence controls; no feature here is a sense."""
from collections import Counter, defaultdict
from fractions import Fraction
import math
import re

from claims.capability_population import need, sha
from text.arithmetic_tasks import equalities


def terms(text):return Counter(re.findall(r'\w+|[^\w\s]',text.casefold()))


def inputs(corpus,key):
    task=corpus.tasks[key];pop=corpus.sources[task['population']].population
    return dict(task=key,kind=task['kind'],group=task['split_group'],fields=[
        {n:pop.fields[f['id']][n] for n in ('id','role','text','clock')}
        for f in task['selections']['context']['fields']])


class ExplicitBaselines:
    def __init__(self,corpus,labels):
        self.corpus=corpus;self.labels=labels;self.bank={};self.index={};self.visits=0
        for key in corpus.train:
            context=inputs(corpus,key);task=corpus.tasks[key];pop=corpus.sources[task['population']].population
            target=pop.fields[task['selections']['targets']['fields'][0]['id']]
            self.bank[key]=dict(context=context,terms=terms('\n'.join(f['text'] for f in context['fields'])),target=target['text'])
        for kind in sorted({r['context']['kind'] for r in self.bank.values()}):
            rows={k:r for k,r in self.bank.items() if r['context']['kind']==kind};df=Counter();postings=defaultdict(list)
            total=sum(sum(r['terms'].values()) for r in rows.values());need(total>0,'BASELINE_EMPTY_DOCUMENTS')
            for key,row in sorted(rows.items()):
                length=sum(row['terms'].values())
                df.update(row['terms'].keys())
                for token,frequency in row['terms'].items():postings[token].append((key,frequency,length))
            counts=Counter(sha(r['target'].encode()) for r in rows.values());modal=min(counts,key=lambda h:(-counts[h],h))
            modal_key=min(k for k,r in rows.items() if sha(r['target'].encode())==modal)
            self.index[kind]=dict(rows=tuple(sorted(rows)),df=df,postings=postings,average=total/len(rows),modal=modal_key)

    def candidate(self,key,method,score=None):
        return dict(method=method,status='candidate',text=self.bank[key]['target'],donor=key,
            region=self.labels[key]['recognition']['label'],score=score,scope='recorded_train_target; donor execution does not transfer')

    def select(self,context):
        kind=context['kind'];need(kind in self.index,'BASELINE_OPERATION')
        index=self.index[kind];need(all(self.bank[k]['context']['group']!=context['group'] for k in index['rows']),'BASELINE_GROUP_LEAK')
        scores=defaultdict(list);count=len(index['rows'])
        for token in sorted(terms('\n'.join(f['text'] for f in context['fields']))):
            weight=math.log1p((count-index['df'][token]+.5)/(index['df'][token]+.5))
            for key,frequency,length in index['postings'].get(token,()):
                self.visits+=1;norm=frequency+1.2*(.25+.75*length/index['average'])
                scores[key].append(weight*frequency*2.2/norm)
        ranked={k:math.fsum(scores[k]) for k in index['rows']};selected=min(ranked,key=lambda k:(-ranked[k],k))
        result={'modal':self.candidate(index['modal'],'modal'),'bm25':self.candidate(selected,'bm25',ranked[selected])}
        applicable=[f for f in context['fields'] if f['role'] in ('recorded_patch','recorded_step','source_answer','attested_speaker_act')]
        if applicable:
            chosen=max(applicable,key=lambda f:(Fraction(f['clock']['end']),f['id'])) if all(f['clock'] and 'end' in f['clock'] for f in applicable) else applicable[-1]
            result['persistence']=dict(method='persistence',status='candidate',text=chosen['text'],field=chosen['id'],region=None,
                scope='compatible input field; equal-clock hash tie is selection only')
        else:result['persistence']=dict(method='persistence',status='unavailable',reason='NO_COMPATIBLE_INPUT_FIELD')
        equations=[r for f in context['fields'] if f['role']=='recorded_step' for r in equalities(f['text'])]
        if kind=='continue_recorded_steps' and equations and equations[-1]['status']=='pass':
            result['last_arithmetic']=dict(method='last_arithmetic',status='candidate',text='#### '+equations[-1]['right_exact'],region=None,
                scope='last input annotation value; no problem interpretation or hidden target-role selection')
        else:result['last_arithmetic']=dict(method='last_arithmetic',status='unavailable',reason='NO_COMPLETED_INPUT_EQUALITY')
        return result
