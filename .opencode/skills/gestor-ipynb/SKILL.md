---
name: gestor-ipynb
description: Activa este skill SIEMPRE que el usuario mencione un archivo .ipynb o pida leer, analizar, editar, agregar, eliminar o ejecutar celdas en un Jupyter Notebook.
---

# Gestión eficiente de Jupyter Notebooks (.ipynb)

## Restricción Crítica
- **NUNCA** uses las herramientas nativas (`read`, `edit`, `write`) directamente sobre un archivo `.ipynb`. Usa siempre el script auxiliar con `py` en Windows (o `python3` en Linux/macOS).
- Siempre que uses `_temp_cell.txt` para `edit` o `add`, **bórralo al terminar** (incluso si el comando falla).

## Comandos disponibles (`ipynb_tool.py`)

1. **Leer el notebook** (haz esto siempre primero para ver los índices; salidas >400 caracteres se truncan):
   `py .opencode/skills/gestor-ipynb/ipynb_tool.py read "<archivo.ipynb>"`

2. **Editar una celda** (guarda el contenido en `_temp_cell.txt` antes de ejecutar):
   `py .opencode/skills/gestor-ipynb/ipynb_tool.py edit "<archivo.ipynb>" <indice> _temp_cell.txt`

3. **Agregar una celda** (`<tipo>` puede ser `code`, `markdown` o `raw`; usa `999` como `<indice>` para insertar al final):
   `py .opencode/skills/gestor-ipynb/ipynb_tool.py add "<archivo.ipynb>" <indice> <tipo> _temp_cell.txt`

4. **Eliminar una celda** (vuelve a ejecutar `read` si harás más cambios, ya que los índices se desplazan):
   `py .opencode/skills/gestor-ipynb/ipynb_tool.py delete "<archivo.ipynb>" <indice>`

5. **Ejecutar todo el notebook** (luego usa `read` para verificar salidas o errores):
   `py .opencode/skills/gestor-ipynb/ipynb_tool.py run "<archivo.ipynb>"`