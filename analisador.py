import os
from google import genai
from google.genai import types
from dataclasses import dataclass
import logging
import argparse
from pydantic import BaseModel


# Configuração do logging para melhor depuração
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Passo 1: Definição do Esquema de Resposta (O Contrato de Dados) ---
@dataclass
class AvaliacaoCompetencia(BaseModel):
    """Estrutura para a avaliação de uma única competência."""
    competencia: str
    pontuacao: int
    justificativa: str

@dataclass
class SugestaoMelhora(BaseModel):
    """Descreve uma sugestão específica para melhorar um trecho da redação."""
    trecho_original: str
    sugestao: str
    explicacao: str

@dataclass
class AnaliseRedacao(BaseModel):
    """Estrutura completa para a análise da redação, conforme o modelo ENEM."""
    analise_geral: str
    pontos_fortes: list[str]
    # CORREÇÃO: Especifique que a lista contém objetos do tipo SugestaoMelhora
    sugestoes_de_melhora: list[SugestaoMelhora]
    avaliacoes_competencias: list[AvaliacaoCompetencia]
    nota_estimada: float

# --- Passo 2: Função Principal de Análise ---
def analisar_redacao(caminho_arquivo: str, tema_redacao: str) -> AnaliseRedacao | None:
    """
    Carrega uma redação, envia para a API Gemini para análise e retorna os resultados estruturados.

    Args:
        caminho_arquivo: O caminho para o arquivo.txt contendo a redação.
        tema_redacao: O tema proposto para a redação.

    Returns:
        Um objeto AnaliseRedacao com os resultados ou None em caso de erro.
    """
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            texto_redacao = f.read()
        logging.info(f"Arquivo '{caminho_arquivo}' lido com sucesso.")
    except FileNotFoundError:
        logging.error(f"Erro: O arquivo '{caminho_arquivo}' não foi encontrado.")
        return None
    except Exception as e:
        logging.error(f"Ocorreu um erro inesperado ao ler o arquivo: {e}")
        return None

    # --- Passo 3: Engenharia de Prompt ---
    prompt = f"""
    Você é um corretor de redações experiente, especialista no modelo de avaliação do ENEM.
    Sua tarefa é analisar a redação a seguir, cujo tema é '{tema_redacao}'.

    Realize uma análise crítica e detalhada, avaliando os seguintes aspectos:

    **1. Competência 1: Domínio da norma padrão da língua escrita (0 a 200 pontos).**
    - Avalia a ausência de desvios gramaticais (concordância, regência, crase) e de convenções de escrita (ortografia, acentuação, pontuação, uso de maiúsculas/minúsculas). Erros graves ou muito frequentes diminuem a nota. [1]

    **2. Competência 2: Compreensão da proposta de redação e aplicação de conceitos (0 a 200 pontos).**
    - Avalia a aderência total ao tema proposto, sem tangenciamentos ou fuga. [1]
    - Avalia o uso de repertório sociocultural pertinente e produtivo (citações, dados, fatos históricos, referências a livros/filmes) que enriqueça a argumentação. [1]

    **3. Competência 3: Selecionar, relacionar, organizar e interpretar argumentos (0 a 200 pontos).**
    - Avalia a clareza da tese (ponto de vista) defendida. [1]
    - Avalia a progressão lógica das ideias e a coerência entre os parágrafos. Os argumentos devem ser bem fundamentados e desenvolvidos, não apenas expostos. [1]

    **4. Competência 4: Demonstração de conhecimento dos mecanismos linguísticos de coesão (0 a 200 pontos).**
    - Avalia o uso adequado de conectivos (conjunções, preposições, pronomes) para ligar orações e parágrafos, garantindo a fluidez e a articulação do texto. [1]

    **5. Competência 5: Elaboração de proposta de intervenção para o problema abordado (0 a 200 pontos).**
    - Avalia a apresentação de uma proposta de intervenção que seja detalhada, exequível e que respeite os direitos humanos. Deve conter obrigatoriamente 5 elementos: Ação (o que fazer?), Agente (quem vai fazer?), Meio/Modo (como será feito?), Efeito (para quê?) e um Detalhamento de um desses elementos. [1]

    Sua resposta DEVE seguir estritamente o schema JSON fornecido. Preencha todos os campos
    de forma completa e objetiva. A nota estimada deve ser um número entre 0 e 1000.

    Para o campo 'sugestoes_de_melhora', gere uma lista de objetos, onde cada objeto deve
    obrigatoriamente conter as chaves 'trecho_original', 'sugestao' e 'explicacao'.

    Para o campo 'avaliacoes_competencias', gere uma lista de objetos, onde cada objeto deve
    obrigatoriamente conter as chaves 'competencia', 'pontuacao' e 'justificativa'.

    Texto da Redação:
    ---
    {texto_redacao}
    ---
    """

    # --- Passo 4: Configuração da Geração Controlada ---
    generation_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=AnaliseRedacao,
        temperature=0.2 # Reduz a aleatoriedade para uma análise mais consistente
    )

    try:
        logging.info("Enviando requisição para a API Gemini...")
        client = genai.Client() # Assume que a chave está na variável de ambiente
        response = client.models.generate_content(
            model="gemini-2.5-flash", # Modelo rápido e capaz para esta tarefa
            contents=prompt,
            config=generation_config
        )

        # --- Passo 5: Acesso Seguro à Resposta Estruturada ---
        analise = response.parsed
        
        if not isinstance(analise, AnaliseRedacao):
             raise TypeError("A resposta da API não corresponde ao schema esperado.")

        logging.info("Análise recebida e processada com sucesso.")
        return analise

    except Exception as e:
        logging.error(f"Ocorreu um erro durante a chamada à API ou processamento da resposta: {e}")
        return None

# --- Passo 6: Função para Exibir os Resultados ---
def exibir_analise(analise: AnaliseRedacao):
    """Formata e exibe a análise da redação no console."""
    print("\n" + "="*50)
    print("      RESULTADO DA ANÁLISE DA REDAÇÃO")
    print("="*50 + "\n")

    print(f"NOTA ESTIMADA: {analise.nota_estimada:.1f} / 1000.0\n")

    print("--- ANÁLISE GERAL ---")
    print(analise.analise_geral + "\n")

    print("--- AVALIAÇÃO POR COMPETÊNCIA ---")
    for comp in analise.avaliacoes_competencias:
        print(f"\n* {comp.competencia}: {comp.pontuacao}/200")
        print(f"  Justificativa: {comp.justificativa}") 

    print("--- PONTOS FORTES ---")
    for ponto in analise.pontos_fortes:
        print(f"- {ponto}")
    print("\n")

    print("--- SUGESTÕES DE MELHORA ---")
    for i, sugestao in enumerate(analise.sugestoes_de_melhora, 1):
        print(f"{i}. Trecho Original: \"{sugestao.trecho_original}\"")
        print(f"   Sugestão: \"{sugestao.sugestao}\"")
        print(f"   Explicação: {sugestao.explicacao}\n")

# --- Ponto de Entrada do Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analisa uma redação usando a API Google GenAI.")
    parser.add_argument("arquivo", type=str, help="Caminho para o arquivo.txt da redação.")
    parser.add_argument("-t", "--tema", type=str, required=True, help="O tema da redação.")
    args = parser.parse_args()
    
    # Verifica se a chave de API está configurada
    if 'GEMINI_API_KEY' not in os.environ:
        logging.error("A variável de ambiente 'GEMINI_API_KEY' não está configurada.")
    else:
        resultado_analise = analisar_redacao(args.arquivo, args.tema)
        if resultado_analise:
            exibir_analise(resultado_analise)