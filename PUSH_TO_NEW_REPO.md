# Как выгрузить `analytics_clean_arch` в отдельный GitHub-репозиторий

Цель: отправить только новый проект `analytics_clean_arch` в отдельный репозиторий:

- `https://github.com/mainarkler/Progect_analitycs`

## Вариант 1 (рекомендуется): готовый скрипт

Из корня текущего репозитория:

```bash
./scripts/push_analytics_clean_arch_to_new_repo.sh https://github.com/mainarkler/Progect_analitycs.git main
```

Скрипт делает:

1. `git subtree split --prefix=analytics_clean_arch` — собирает историю только для подпроекта.
2. Пушит результат в новый удалённый репозиторий как отдельный проект.
3. Чистит временные сущности (branch + remote).

## Вариант 2 (вручную)

```bash
git subtree split --prefix=analytics_clean_arch -b tmp/analytics_clean_arch_split
git push https://github.com/mainarkler/Progect_analitycs.git tmp/analytics_clean_arch_split:main
git branch -D tmp/analytics_clean_arch_split
```

## Если push не проходит

Обычно проблема в авторизации/правах:

- Убедитесь, что вы залогинены (`gh auth login`) или используете PAT.
- Проверьте, что у вас есть права записи в `mainarkler/Progect_analitycs`.
- При необходимости пушьте через SSH URL.
