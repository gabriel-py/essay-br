## 🤖 Avaliação de LLMs para Correção Automática de Redações ENEM (Gemini & Prompt Engineering)

Este projeto foi desenvolvido no contexto da disciplina **Resolução de Problemas II** da EACH-USP e tem como objetivo investigar o uso de **Modelos de Linguagem de Grande Porte (LLMs)**, em especial o **Gemini da Google**, para a correção automática de redações do ENEM.

### 🎯 Objetivo e Motivação

O processo manual de correção de redações é caro, demorado e sujeito a divergências. A motivação central é avaliar se modelos de IA podem oferecer uma alternativa mais **rápida, escalável e consistente**.

O foco específico deste experimento foi avaliar como diferentes técnicas de **Engenharia de Prompt** influenciam a qualidade das correções produzidas pelo Gemini, utilizando o **dataset Essay-BR** (redações reais com notas por competência).

---

### ⚙️ Experimento: Variações de Prompting

Foram construídas e testadas três variações de prompt base para corrigir as 100 primeiras redações do *Essay-BR*, analisando seu impacto na acurácia e concordância com corretores humanos.

| Variação de Prompt | Técnica | Descrição |
| :--- | :--- | :--- |
| **Prompt 1** | Zero-shot | Apenas instruções diretas, sem exemplos ou raciocínio aprofundado. |
| **Prompt 2** | Few-shot | Fornece **exemplos** completos de correções, orientando o padrão de saída. |
| **Prompt 3** | Chain-of-Thought (CoT) | Orienta o modelo a seguir um **raciocínio avaliativo detalhado e estruturado** (Competências ENEM) antes de emitir a nota. |

---

### 💻 Pipeline Experimental

O processo experimental está organizado em uma pipeline de três scripts principais:

| Script | Função | Exemplo de Uso |
| :--- | :--- | :--- |
| `analisador.py` | Envia as redações ao Gemini, montando o prompt final e gravando o resultado bruto (`resultado_ia` em JSON). | `python analisador.py --in essay-br.csv --prompt-file prompt1.txt --n 100` |
| `gemini_normalizer.py` | Faz o *parse* do resultado bruto da IA (JSON) e transforma as notas previstas e as notas reais em colunas numéricas organizadas. | Recebe o CSV gerado pelo `analisador.py`. |
| `metrics.py` | Calcula as métricas de desempenho (**MAE** e **QWK**) de forma agregada e por tema de redação. | Recebe o CSV normalizado. |

> 🔑 **Pré-requisito:** A variável de ambiente `GEMINI_API_KEY` deve estar definida com uma chave de acesso válida.

---

### 📊 Resultados e Conclusões

Os resultados consolidados mostram o impacto direto da engenharia de prompt na performance do modelo:

#### 📈 Métricas Consolidadas

| Técnica de Prompt | MAE (Mean Absolute Error) | QWK (Quadratic Weighted Kappa) |
| :--- | :--- | :--- |
| **1. Zero-shot** | 52,8 | 0,876 |
| **2. Few-shot** | 36,8 | 0,9005 |
| **3. Chain-of-Thought (CoT)** | **32,8 (Menor Erro)** | **0,9415 (Maior Concordância)** |

#### ✅ Conclusões

1.  **Prompts Simples** (`Zero-shot`) não são suficientes para atingir alta acurácia em tarefas complexas.
2.  A técnica **Few-shot** melhora a performance ao fornecer exemplos concretos, reduzindo o erro e aumentando a concordância com avaliadores humanos.
3.  A técnica **Chain-of-Thought (CoT)** alcançou o melhor desempenho, com um QWK compatível com o nível de concordância entre corretores humanos especializados. Isso sugere que instruir o modelo a raciocinar de forma estruturada torna a avaliação mais estável e alinhada aos critérios do ENEM.

> **Observação de Custos:** Prompts mais complexos (Few-shot e CoT) tendem a consumir mais tokens, aumentando o custo de uso do modelo em larga escala. É necessário um equilíbrio entre qualidade e viabilidade econômica.

---

### 🚀 Trabalhos Futuros

* **Testar Prompts Híbridos:** Combinar as técnicas few-shot e CoT para buscar resultados ainda mais assertivos.
* **Expansão da Análise:** Estender os experimentos para um conjunto maior e mais diversificado de redações.
* **Análise Fina:** Aprofundar as análises por competência para identificar pontos fortes e fracos do modelo.

O projeto demonstra que a engenharia de prompt é um **componente essencial** para transformar um LLM genérico (como o Gemini) em uma ferramenta específica e confiável para avaliação educacional.
