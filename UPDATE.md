


## Merge cambios de la rama principal. 

Ver si hay cambios
* https://github.com/Mr-McGL-Garage/VibeVoice

--

* https://github.com/microsoft/VibeVoice
* https://github.com/microsoft/VibeVoice/network
* https://github.com/Mr-McGL-Garage/VibeVoice/network

```bash
git remote -v 


# Sino esta
git remote add upstream https://github.com/Mr-McGL-Garage/VibeVoice.git

git switch -c merge-upstream-main-1
git fetch upstream

# Comparación
git status
git log --oneline --left-right --graph HEAD...upstream/main
git diff --stat HEAD..upstream/main

git merge upstream/main

# Resuelve los conflictos, si los hay, y luego haz commit de la fusión.

git push -u origin merge-upstream-main-1
```

## Cambio 1:

Encontrar la función que falla, desde al carpeta padre del proyecto:
```bash
grep -R "_prepare_cache_for_generation(" -n vibevoice
```

Creo que este código funciona `vibevoice/vibevoice/modular/modeling_vibevoice_streaming_inference.py`:

```py
        try:
            from transformers.cache_utils import DynamicCache
            sig = inspect.signature(DynamicCache.__init__)
            if 'config' in sig.parameters:
                # transformers >= 4.57: let model handle cache creation
                return None
            else:
                # Older versions: use parent method <--- self._prepare_cache_for_generation_compat(generation_config,model_kwargs,None,batch_size,max_cache_length,device)
                prep_sig = inspect.signature(self._prepare_cache_for_generation)
                if 'device' in prep_sig.parameters:
                    self._prepare_cache_for_generation(generation_config, model_kwargs, None, batch_size, max_cache_length, device)
                else:
                    self._prepare_cache_for_generation(generation_config, model_kwargs, None, batch_size, max_cache_length)
                return model_kwargs.get("past_key_values")
        except Exception:
            return None
```

Funciones corregidas:
* `vibevoice/vibevoice/modular/modeling_vibevoice_inference.py` 

Cambios:

```py
######################################
import inspect
######################################
```

```py

```



No tocar, funciones antiguas y ya corregidas:
* `vibevoice/vibevoice/modular_old/modeling_vibevoice_inference.py` -- No tocar
* `vibevoice/vibevoice/modular_old/modeling_vibevoice_streaming_inference.py` -- No tocar
* `vibevoice/vibevoice/modular/modeling_vibevoice_streaming_inference.py` -- No tocar



