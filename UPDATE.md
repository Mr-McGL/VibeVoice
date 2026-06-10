
Ver si hay cambios
https://github.com/Mr-McGL-Garage/VibeVoice/network

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

