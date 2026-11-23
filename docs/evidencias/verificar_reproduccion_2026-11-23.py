"""Ejecuta el entrenamiento completo en una carpeta aislada y compara métricas."""
from pathlib import Path
import sys,json,shutil,hashlib,platform,traceback

root=Path.cwd()
sys.path.insert(0,str(root/'scripts'))
import entrenar_modelo as entrenamiento

out=root/'.build_defensa/reproduccion_2026-11-23'
out.mkdir(parents=True,exist_ok=True)
shutil.copy2(root/'models/mejores_hiperparametros.json',out/'mejores_hiperparametros.json')
before={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ('models/metricas.json','models/mejor_modelo_lgbm.joblib','index.html')}
entrenamiento.RUTA_MODELOS=out
result={'fecha':'2026-11-23','python':platform.python_version(),'comando':'.venv/Scripts/python.exe .build_defensa/verificar_reproduccion.py','protocolo':'Script completo: baseline y mejorado con GroupKFold(5), evaluación temporal, selección OOF de umbrales y entrenamiento final. Salida aislada de los artefactos vigentes.','versiones':{m:__import__(m).__version__ for m in ('pandas','numpy','scipy','sklearn','lightgbm','joblib')}}
try:
    entrenamiento.main()
    actual=json.loads((out/'metricas.json').read_text(encoding='utf-8'))
    expected=json.loads((root/'models/metricas.json').read_text(encoding='utf-8'))
    differences=[]
    def compare(a,b,p=''):
        for k,v in a.items():
            if k=='generado_en':continue
            q=f'{p}.{k}' if p else k
            if isinstance(v,dict):compare(v,b[k],q)
            elif isinstance(v,(int,float)):
                delta=abs(v-b[k])
                if delta>1e-9:differences.append({'metrica':q,'vigente':v,'reproducido':b[k],'diferencia':delta})
            elif v!=b[k]:differences.append({'campo':q,'vigente':v,'reproducido':b[k]})
    compare(expected,actual)
    after={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in before}
    result.update({'estado':'reproducido' if not differences else 'ejecutado_con_diferencias','tolerancia_absoluta':1e-9,'diferencias':differences,'metricas_reproducidas':actual,'artefactos_vigentes_sin_cambios':before==after,'huellas_vigentes':after})
except BaseException as e:
    result.update({'estado':'fallo','error':str(e),'detalle':traceback.format_exc()})
finally:
    dest=root/'docs/evidencias/reproduccion_entrenamiento_2026-11-23.json'
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('VERIFICACION_FINAL',result['estado'],flush=True)
    print('EVIDENCIA',dest,flush=True)
if result['estado']!='reproducido':raise SystemExit(1)
