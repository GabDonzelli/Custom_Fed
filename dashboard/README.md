# Painel de resultados

Desenha as simulações federadas deste projeto a partir dos CSVs que o
`ResultsLogger` escreve. Lê os dados, nunca os altera.

```bash
cd dashboard
python servidor.py     # ao vivo em http://127.0.0.1:8771/ , relê os CSVs a cada 3s
python congelar.py     # gera os HTML autocontidos, sem servidor
```

Sem dependências além da biblioteca padrão do Python.

## Arquivos

| arquivo | papel |
|---|---|
| `index.html` | a página: barra lateral, gráficos, tabelas, achados. Bilíngue pt/en |
| `simulacoes.json` | o manifesto — quais simulações existem, quais CSVs, a ficha técnica |
| `servidor.py` | serve a página e o endpoint `/api/data` |
| `congelar.py` | embute os dados na página e grava os HTML offline |
| `traduzir.py` | (re)escreve os campos `_en` e a seção de achados no manifesto |
| `renomear.py` | aplica o esquema de nomes das simulações |
| `dados/` | os CSVs lidos pelo painel |

`servidor.py` relê o manifesto e os CSVs a cada requisição, e `index.html` é
lido do disco a cada requisição também — editar a página e dar F5 basta, sem
reiniciar nada.

## O que o `congelar.py` gera

Quatro arquivos, todos autocontidos (dados embutidos, funcionam offline, sem
servidor) e todos bilíngues — o botão de idioma alterna em qualquer um:

- `../resultados.html` e `../results.html` — na raiz do projeto, versionados
- `Resultados.html` e `Results.html` no vault do Obsidian, **se a pasta
  existir**. Num clone sem vault o script simplesmente pula esses dois.

Ele valida cinco âncoras de texto no `index.html` antes de gerar e **falha em
vez de produzir uma página quebrada em silêncio**. Se você mexer no bloco de
`tick()`, ajuste as âncoras no topo do `congelar.py` junto.

## Acrescentar uma simulação

Editar `simulacoes.json`, acrescentar uma entrada em `simulacoes`, e rodar
`python congelar.py`. Só `config.clients` e `config.rounds` são obrigatórios —
o eixo dos gráficos e as barras de progresso dependem deles. O resto é
opcional e só aparece se estiver lá.

Runs são casadas **por `id`** entre simulações: é assim que a linha tracejada
de referência encontra a tarefa correspondente na outra simulação.

## Caminhos

`project_dir` no manifesto é resolvido a partir da pasta do próprio manifesto,
então o valor `"dados"` funciona em qualquer clone. Um caminho absoluto também
é aceito, se algum dia os CSVs morarem fora do repositório.

Atenção a uma armadilha do Git: os CSVs são versionados, então um
`git switch` para uma branch que não os tem **apaga-os da pasta** e o painel
fica vazio. Eles voltam ao trocar de volta — nada se perde, mas o susto é real.
