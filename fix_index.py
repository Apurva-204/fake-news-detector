import re

filepath = 'templates/index.html'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

replacements = {
    'background: linear-gradient(160deg, #F5F5F5 0%, #EAEEF3 50%, #F0F0F5 100%);': 'background: var(--bg-color);',
    'background: #FFFFFF;': 'background: var(--card-bg);\n      border: 1px solid var(--border-color);',
    'color: #111;': 'color: var(--text-primary);',
    'color: #777;': 'color: var(--text-secondary);',
    'background: #F7F7F9;': 'background: var(--bg-color);',
    'background: #FFF;': 'background: var(--card-bg);',
    'color: #AAA;': 'color: var(--text-muted);',
    'border: 2px dashed #DDD;': 'border: 2px dashed var(--border-color);',
    'background: #FAFAFE;': 'background: var(--upload-bg);',
    'background: #F5F5FF;': 'background: var(--upload-drag-bg);',
    'background: #F0FDF4;': 'background: var(--tag-real-bg);',
    'color: #FFF;': 'color: var(--bg-color);',
    'background: #F5F5F5;': 'background: var(--bg-color);',
    'background: #EEE;': 'background: var(--hover-bg);',
    'border-top: 1px solid #EEE;': 'border-top: 1px solid var(--border-color);',
    'color: #16A34A;': 'color: var(--color-real);',
    'color: #DC2626;': 'color: var(--color-fake);',
    'background: #F0F0F0;': 'background: var(--border-alpha);',
    'color: #555;': 'color: var(--text-secondary);',
    'color: #999;': 'color: var(--text-muted);',
    'color:#111;': 'color:var(--text-primary);',
    'color:#777;': 'color:var(--text-secondary);',
    'color:#FFF;': 'color:var(--card-bg);',
}

for k, v in replacements.items():
    content = content.replace(k, v)

# Fix verify button
content = content.replace('background: linear-gradient(135deg, #6366F1, #8B5CF6);', 'background: var(--text-primary); color: var(--card-bg);')
content = content.replace('color: var(--bg-color);', 'color: var(--card-bg);') # Some cases it might be better as card-bg

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
