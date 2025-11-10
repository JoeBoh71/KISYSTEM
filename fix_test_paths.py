from pathlib import Path

file_path = Path('tests/test_hybrid_handler.py')

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the paths
old_paths = """sys.path.insert(0, str(Path(__file__).parent / 'core'))
sys.path.insert(0, str(Path(__file__).parent / 'agents'))"""

new_paths = """sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'agents'))"""

if old_paths in content:
    content = content.replace(old_paths, new_paths)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('✅ Path fix applied!')
else:
    print('⚠️  Pattern not found, checking...')
    print('Looking for:', repr(old_paths[:50]))
    
