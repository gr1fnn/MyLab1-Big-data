import sys

print("Testing FLAML import...")
print(f"Python: {sys.executable}")

try:
    import flaml
    print(f"FLAML version: {flaml.__version__}")
    print(f"FLAML location: {flaml.__file__}")
    print(f"Available in flaml module: {[x for x in dir(flaml) if not x.startswith('_')]}")
except Exception as e:
    print(f"Error importing flaml: {e}")

try:
    from flaml import AutoML
    print("from flaml import AutoML - SUCCESS")
except Exception as e:
    print(f"from flaml import AutoML - FAILED: {e}")

try:
    from flaml import FLAML
    print("from flaml import FLAML - SUCCESS")
except Exception as e:
    print(f"from flaml import FLAML - FAILED: {e}")

try:
    import flaml.automl
    print("import flaml.automl - SUCCESS")
    print(f"flaml.automl contents: {dir(flaml.automl)}")
except Exception as e:
    print(f"import flaml.automl - FAILED: {e}")

#pip install flaml[automl]
#streamlit run app.py
#python test.py      