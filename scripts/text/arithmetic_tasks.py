"""Attested arithmetic steps with exact, bounded, non-executing equality checks."""
import argparse
import ast
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text.capability_tasks import TaskBuilder, check, text_sha
from text.evaluation_multiscale import file_sha
from text.gsm8k_source import FILES, REVISION, git_blob


def exact(expression):
    """Only decimal/rational arithmetic. No eval, calls, names or floating approximation."""
    check(isinstance(expression, str) and 0 < len(expression) <= 256, 'ARITHMETIC_SIZE')
    tree = ast.parse(expression, mode='eval')
    check(sum(1 for _ in ast.walk(tree)) <= 64, 'ARITHMETIC_NODES')

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            literal = ast.get_source_segment(expression, node)
            check(bool(re.fullmatch(r'(?:\d+(?:\.\d*)?|\.\d+)', literal)), 'ARITHMETIC_LITERAL')
            out = Fraction(literal)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
            out = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): out = a + b
            elif isinstance(node.op, ast.Sub): out = a - b
            elif isinstance(node.op, ast.Mult): out = a * b
            else: out = a / b
        else:
            raise ValueError('MNLM_CAPABILITY_TASK_ARITHMETIC_OPERATOR')
        check(out.numerator.bit_length() <= 1024 and out.denominator.bit_length() <= 1024,
              'ARITHMETIC_MAGNITUDE')
        return out
    return visit(tree.body)


def equalities(text):
    records = []
    for match in re.finditer(r'<<([^<>]+)>>', text):
        expression = match[1]
        record = dict(annotation=expression, codepoint_span=list(match.span()),
                      scope='annotated_equality_only; not word-problem interpretation')
        try:
            check(expression.count('=') == 1, 'ARITHMETIC_EQUALITY')
            left, right = expression.split('=')
            a, b = exact(left.strip()), exact(right.strip())
            record.update(left_exact=str(a), right_exact=str(b), status='pass' if a == b else 'fail')
        except (ValueError, SyntaxError, ZeroDivisionError, RecursionError) as exc:
            record.update(status='unsupported', error=str(exc))
        records.append(record)
    return records


def build(source, output, count):
    check(type(count) is int and 1 <= count <= 96, 'ARITHMETIC_PROOF_CORPUS_BOUND')
    start = time.perf_counter()
    pins = {}
    for name, (_, size, blob) in FILES.items():
        data = (source / name).read_bytes()
        check(len(data) == size and git_blob(data) == blob, 'ARITHMETIC_SOURCE_PIN')
        pins[name] = file_sha(source / name)
    raw = (source / 'train.jsonl').read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    source_record = dict(repository='openai/grade-school-math', revision=REVISION, files=pins,
        licence='MIT', attribution='OpenAI, 2021', encoder_exposure_certified=False,
        selection=dict(method='first-source-rows-before-check-results', count=count, total=len(rows)),
        producer_sha256=file_sha(Path(__file__)))
    out = TaskBuilder('reasoning', source_record)
    checks, step_counts = Counter(), []
    for index, row in enumerate(rows[:count]):
        address = dict(artifact_sha256=pins['train.jsonl'], row=index)
        group = 'gsm8k-question:' + text_sha(row['question'])
        problem = out.field(row['question'], 'problem', dict(address, field='question'))
        answer = out.field(row['answer'], 'recorded_solution', dict(address, field='answer'))
        out.edge(problem, answer, 'provider_solution', address)
        records = equalities(row['answer'])
        checks.update(r['status'] for r in records)
        out.task(context=[problem], targets=[answer], consequences=[], group=group,
                 kind='solve_recorded_problem', provenance=address, checks=records)
        previous, offset = [], 0
        for line in row['answer'].splitlines(keepends=True):
            text = line.rstrip('\r\n')
            check(bool(text.strip()), 'ARITHMETIC_BLANK_STEP')
            role = 'recorded_conclusion' if text.startswith('#### ') else 'recorded_step'
            step = out.field(text, role, dict(address, field='answer',
                                             codepoint_span=[offset, offset + len(text)]))
            out.edge(answer, step, 'contains', address)
            out.edge(previous[-1] if previous else problem, step, 'source_sequence', address)
            out.task(context=[problem, *previous], targets=[step], consequences=[], group=group,
                     kind='continue_recorded_steps', provenance=dict(address, position=len(previous)),
                     checks=equalities(text))
            previous.append(step)
            offset += len(line)
        check(offset == len(row['answer']), 'ARITHMETIC_STEP_CONSERVATION')
        step_counts.append(len(previous))
        if not records or any(r['status'] != 'pass' for r in records):
            out.gaps.append(dict(row=index, reason='missing_failed_or_unsupported_arithmetic',
                                 records=records, source_retained=True))
    report = out.publish(output, scope='source-stated arithmetic derivations; exact annotated equalities only',
        coverage=dict(problems=count, source_steps=sum(step_counts), annotated_equalities=dict(checks),
                      conclusion_interpretation_checked=False, revisions_present=False,
                      split_assignment='not yet learning eligible; task families must stay grouped'))
    print(json.dumps(dict(report, wall_seconds=time.perf_counter() - start)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=96)
    args = parser.parse_args()
    build(args.source, args.output, args.count)
