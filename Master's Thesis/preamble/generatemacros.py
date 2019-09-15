import string
import argparse
import os
import IPython


# Define list of greek letters avaiable in LaTeX
greek_alphabet = [
        'alpha',
        'theta',
        'tau',
        'beta',
        'vartheta',
        'pi',
        'upsilon',
        'gamma',
        'gamma',
        'varpi',
        'phi',
        'delta',
        'kappa',
        'rho',
        'varphi',
        'epsilon',
        'lambda',
        'varrho',
        'chi',
        'varepsilon',
        'mu',
        'sigma',
        'psi',
        'zeta',
        'nu',
        'varsigma',
        'omega',
        'eta',
        'xi',
        'Gamma',
        'Lambda',
        'Sigma',
        'Psi',
        'Delta',
        'Xi',
        'Upsilon',
        'Omega',
        'Theta',
        'Pi',
        'Phi']


def define_command(name, cmd, n_inputs=0):
    if n_inputs == 0:
        s = '\\providecommand{\\' + name + '}{{' + cmd + '}}\n'
        s += '\\renewcommand{\\' + name + '}{{' + cmd + '}}\n'
    else:
        s = '\\providecommand{\\' + name + '}[' + str(n_inputs) + ']{{' + cmd + '}}\n'
        s += '\\renewcommand{\\' + name + '}[' + str(n_inputs) + ']{{' + cmd + '}}\n'
    return s


def various():
    s = '% Parenthesis\n'
    s += define_command('pa', '\\left(#1\\right)', 1)
    s += define_command('bra', '\\left[#1\\right]', 1)
    s += define_command('cbra', '\\left\\{#1\\right\\}', 1)
    s += define_command('vbra', '\\left\\langle#1\\right\\rangle', 1)
    s += '\n% Matrices for displayed expressions\n'
    s += define_command('mat', '\\begin{matrix}#1\\end{matrix}', 1)
    s += define_command('pmat', '\\begin{pmatrix}#1\\end{pmatrix}', 1)
    s += define_command('bmat', '\\begin{bmatrix}#1\\end{bmatrix}', 1)
    s += '\n% Variations of \\frac and \\sfrac\n'
    s += define_command('pfrac', '\\left(\\frac{#1}{#2}\\right)', 2)
    s += define_command('bfrac', '\\left[\\frac{#1}{#2}\\right]', 2)
    s += define_command('psfrac', '\\left(\\sfrac{#1}{#2}\\right)', 2)
    s += define_command('bsfrac', '\\left[\\sfrac{#1}{#2}\\right]', 2)
    s += '\n% for small matrices to be used in in-line expressions\n'
    s += define_command('sm', '\\left\\{#1\\right\\}', 1)
    s += define_command('psm', '\\pa{\\sm{#1}}', 1)
    s += define_command('bsm', '\\bra{\\sm{#1}}', 1)
    s += '\n% Norm\n'
    s += define_command('norm', '\\left\\lVert#1\\right\\rVert', 1)
    s += '% Size\n'
    s += define_command('size', '\\left\\lvert#1\\right\\rvert', 1)
    s += '% Trace\n'
    s += define_command('Tr', '\\text{Tr}\\left[#1\\right]', 1)
    s += '% Tranpose\n'
    s += define_command('transpose', '^\\mathrm{T}')
    s += '\n% Derivatives\n'
    s += define_command('od', '\\frac{\\text{d}^{#3}#1}{\\text{d}^{#3}#2}', 3)
    s += define_command('pd', '\\frac{\\partial^{#3}#1}{\\partial^{#3}#2}', 3)
    return s + '\n'


def boldslanted():
    s = '% Bold letters\n'
    for c in string.ascii_letters:
        s += define_command(c + 'bs', '{\\boldsymbol{' + c + '}' + '}')
    # s += '% Bold numbers\n'
    # for c in string.digits:
    #     s += define_command(c + 'bs', '{\\boldsymbol{' + c + '}' + '}')
    s += '% Bold greek symbols\n'
    for c in greek_alphabet:
        s += define_command(c + 'bs', '{\\boldsymbol{\\' + c + '}' + '}')
    return s + '\n'


def boldupright():
    s = '% Bold upright letters\n'
    for c in string.ascii_letters:
        s += define_command(c + '', '\\mathbf{' + c + '}')
    s += '% Bold upright numbers\n'
    for c in string.digits:
        s += define_command(c, '\\mathbf{' + c + '}')
    s += '% Bold upright greek symbols\n'
    for c in greek_alphabet:
        if c[0].isupper():
            s += define_command(c + 'b', '\\boldsymbol{\\' + 'Up' + c.lower() + '}')
        else:
            s += define_command(c + 'b', '\\boldsymbol{\\' + 'up' + c + '}')
    return s + '\n'


def mathbb():
    s = '% \\mathbb{} shortcuts\n'
    for c in string.ascii_letters:
        s += define_command(c + 'bb', '\\mathbb{' + c + '}')
    return s + '\n'


def mathcal():
    s = '% \\mathcal{} shortcuts\n'
    for c in string.ascii_letters:
        s += define_command(c + 'c', '\\mathcal{' + c + '}')
    return s + '\n'


def writetofile(file, string):
    with open(file, 'w') as f:
        f.write(string)


def generatemacros(name):
    s = '%!TEX root = main.tex\n\n'
    s += various()
    s += boldupright()
    s += boldslanted()
    s += mathbb()
    s += mathcal()
    path = os.path.dirname(os.path.realpath(__file__))
    name = name + '.tex'
    f = os.path.join(path, name)
    writetofile(f, s)


if __name__ == '__main__':
    # Parse input
    parser = argparse.ArgumentParser(description='Generate a LaTeX macros file.')
    parser.add_argument('-n', metavar='--name', default='macros', type=str, 
                        help='The name of the macros file')
    args = parser.parse_args()
    
    # Genearte macro
    generatemacros(args.n)








