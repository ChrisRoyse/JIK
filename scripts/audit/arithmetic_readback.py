"""Independent token/precedence evaluator for retained arithmetic-check receipts."""
from fractions import Fraction
import re

from claims.capability_population import need


def value(expression):
    need(0 < len(expression) <= 256, 'READBACK_ARITHMETIC_SIZE')
    tokens = re.findall(r'(?:\d+(?:\.\d*)?|\.\d+)|[()+*/-]', expression)
    need(''.join(tokens) == re.sub(r'\s', '', expression) and len(tokens) <= 64,
         'READBACK_ARITHMETIC_TOKENS')
    position = 0

    def parse(minimum=0):
        nonlocal position
        need(position < len(tokens), 'READBACK_ARITHMETIC_OPERAND')
        token = tokens[position]
        position += 1
        if token in ('+', '-'):
            left = parse(3) * (-1 if token == '-' else 1)
        elif token == '(':
            left = parse()
            need(position < len(tokens) and tokens[position] == ')', 'READBACK_ARITHMETIC_PARENS')
            position += 1
        else:
            need(bool(re.fullmatch(r'(?:\d+(?:\.\d*)?|\.\d+)', token)), 'READBACK_ARITHMETIC_NUMBER')
            left = Fraction(token)
        while position < len(tokens) and tokens[position] != ')':
            op = tokens[position]
            priority = {'+': 1, '-': 1, '*': 2, '/': 2}.get(op, -1)
            need(priority >= 0, 'READBACK_ARITHMETIC_OPERATOR')
            if priority < minimum: break
            position += 1
            right = parse(priority + 1)
            if op == '+': left += right
            elif op == '-': left -= right
            elif op == '*': left *= right
            else: left /= right
            need(left.numerator.bit_length() <= 1024 and left.denominator.bit_length() <= 1024,
                 'READBACK_ARITHMETIC_MAGNITUDE')
        return left

    result = parse()
    need(position == len(tokens), 'READBACK_ARITHMETIC_TRAILING')
    return result


def receipt(text, records):
    annotations = list(re.finditer(r'<<([^<>]+)>>', text))
    need(len(annotations) == len(records), 'READBACK_ARITHMETIC_COVERAGE')
    for match, record in zip(annotations, records):
        need(record['annotation'] == match[1] and record['codepoint_span'] == list(match.span()),
             'READBACK_ARITHMETIC_ADDRESS')
        try:
            sides = match[1].split('=')
            need(len(sides) == 2, 'READBACK_ARITHMETIC_EQUALITY')
            left, right = map(value, sides)
        except (ValueError, ZeroDivisionError, RecursionError):
            need(record['status'] == 'unsupported', 'READBACK_ARITHMETIC_UNSUPPORTED')
        else:
            need(record['left_exact'] == str(left) and record['right_exact'] == str(right)
                 and record['status'] == ('pass' if left == right else 'fail'), 'READBACK_ARITHMETIC_VALUE')
