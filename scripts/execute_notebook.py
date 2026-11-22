"""Execute a notebook in a fresh kernel using this Python interpreter."""
import argparse
from pathlib import Path
import sys
import tempfile
import os


def execute_notebook(path, output=None):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    path = Path(path).resolve()
    notebook = nbformat.read(path, as_version=4)
    with tempfile.TemporaryDirectory(prefix='cql-ipython-') as directory:
        options = {'transport': 'ipc', 'ip': str(Path(directory) / 'ipc')} if os.name == 'posix' else {}
        manager = KernelManager(kernel_name='python3', connection_file=str(Path(directory) / 'kernel.json'), **options)
        manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
        client = NotebookClient(notebook, km=manager, timeout=120, startup_timeout=30, record_timing=False,
                                resources={'metadata': {'path': str(path.parent)}})
        try:
            client.execute(cleanup_kc=True, env={**os.environ, 'IPYTHONDIR': directory})
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
    nbformat.validate(notebook)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, output)
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', type=Path)
    parser.add_argument('--output', type=Path, help='Save executed outputs; otherwise leave the input unchanged')
    args = parser.parse_args()
    notebook = execute_notebook(args.notebook, args.output)
    count = sum(cell.cell_type == 'code' for cell in notebook.cells)
    print(f'Executed {count} code cells successfully in a fresh kernel.')


if __name__ == '__main__':
    main()
