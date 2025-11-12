# Quick fix for unicode encoding in run_u3daw_autonomous.py
import sys

input_file = "run_u3daw_autonomous.py"
output_file = "run_u3daw_autonomous_fixed.py"

with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all .write_text( with .write_text(encoding='utf-8',
content = content.replace('.write_text(', '.write_text(encoding="utf-8", ')

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"✅ Fixed: {output_file}")
print("Run with: python run_u3daw_autonomous_fixed.py")