import json
import subprocess
import sys


def load_nb(path):
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)


def save_nb(path, nb):
  with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")


def to_source_lines(text):
  lines = text.splitlines(keepends=True)
  return lines if lines else []


def read_nb(path):
  nb = load_nb(path)
  for idx, cell in enumerate(nb.get("cells", [])):
    ctype = cell.get("cell_type")
    source = "".join(cell.get("source", [])).strip()
    print(f"=== CELDA {idx} [{ctype.upper()}] ===")
    print(source if source else "(vacía)")

    if ctype == "code":
      for out in cell.get("outputs", []):
        if out.get("output_type") == "error":
          print(f"\n[ERROR]: {out.get('ename')}: {out.get('evalue')}")
          continue
        text = out.get("text") or out.get("data", {}).get("text/plain")
        if text:
          clean = "".join(text) if isinstance(text, list) else str(text)
          if len(clean) > 400:
            clean = clean[:400] + "\n... [TRUNCADO]"
          print(f"\n[Output]:\n{clean}")
    print()


def edit_cell(path, idx, new_code):
  nb = load_nb(path)
  cells = nb.get("cells", [])
  if idx < 0 or idx >= len(cells):
    print(f"Error: Índice {idx} fuera de rango.")
    sys.exit(1)
  cells[idx]["source"] = to_source_lines(new_code)
  if cells[idx]["cell_type"] == "code":
    cells[idx]["outputs"] = []
    cells[idx]["execution_count"] = None
  save_nb(path, nb)
  print(f"OK: Celda {idx} actualizada.")


def add_cell(path, idx, ctype, new_code):
  nb = load_nb(path)
  cells = nb.get("cells", [])
  new_cell = {
      "cell_type": ctype,
      "metadata": {},
      "source": to_source_lines(new_code),
  }
  if ctype == "code":
    new_cell["outputs"] = []
    new_cell["execution_count"] = None
  idx = max(0, min(idx, len(cells)))
  cells.insert(idx, new_cell)
  save_nb(path, nb)
  print(f"OK: Celda {ctype} insertada en {idx}.")


def delete_cell(path, idx):
  nb = load_nb(path)
  cells = nb.get("cells", [])
  if idx < 0 or idx >= len(cells):
    print(f"Error: Índice {idx} fuera de rango.")
    sys.exit(1)
  cells.pop(idx)
  save_nb(path, nb)
  print(f"OK: Celda {idx} eliminada.")


def run_nb(path):
  try:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    with open(path, 'r', encoding='utf-8') as f:
      nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=900)
    ep.preprocess(nb, {'metadata': {'path': '.'}})
    with open(path, 'w', encoding='utf-8') as f:
      nbformat.write(nb, f)
    print("OK: Notebook ejecutado correctamente. Usa 'read' para ver los resultados.")
  except Exception as e:
    print(f"Error al ejecutar:\n{e}")


if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("Uso: ipynb_tool.py <read|edit|add|delete|run> <archivo.ipynb> [args...]")
    sys.exit(1)

  action = sys.argv[1]
  file_path = sys.argv[2]

  try:
    if action == "read":
      read_nb(file_path)
    elif action == "edit":
      with open(sys.argv[4], "r", encoding="utf-8") as f:
        content = f.read()
      edit_cell(file_path, int(sys.argv[3]), content)
    elif action == "add":
      ctype = sys.argv[4]
      if ctype not in ("code", "markdown", "raw"):
        print("Error: El tipo de celda debe ser 'code', 'markdown' o 'raw'.")
        sys.exit(1)
      with open(sys.argv[5], "r", encoding="utf-8") as f:
        content = f.read()
      add_cell(file_path, int(sys.argv[3]), ctype, content)
    elif action == "delete":
      delete_cell(file_path, int(sys.argv[3]))
    elif action == "run":
      run_nb(file_path)
    else:
      print(f"Acción desconocida: {action}")
      sys.exit(1)
  except IndexError:
    print("Error: Faltan argumentos para la acción solicitada.")
    sys.exit(1)
