# PyWhisper 

## Índice

- [Visão Geral](#visão-geral)
- [Recursos](#recursos)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Uso](#uso)
  - [Transcrever um Único Arquivo de Áudio](#transcrever-um-único-arquivo-de-áudio)
  - [Transcrever Múltiplos Arquivos de Áudio em um Diretório](#transcrever-múltiplos-arquivos-de-áudio-em-um-diretório)
- [Logs](#logs)
- [Otimização de Desempenho](#otimização-de-desempenho)
- [Resolução de Problemas](#resolução-de-problemas)
- [Contribuindo](#contribuindo)
- [Licença](#licença)
- [Agradecimentos](#agradecimentos)

## Visão Geral

O **Pipeline de Transcrição de Áudio** é uma ferramenta baseada em Python desenvolvida para transcrever arquivos de áudio em texto utilizando o modelo [Whisper](https://github.com/openai/whisper) da Hugging Face. Ele suporta processamento paralelo para lidar de forma eficiente com múltiplos chunks e arquivos de áudio simultaneamente, aproveitando multi-threading para otimizar o desempenho.

## Recursos

- **Reconhecimento Automático de Fala (ASR):** Converte conteúdo de áudio em texto escrito usando modelos avançados de aprendizado de máquina.
- **Processamento Paralelo:** Utiliza múltiplas threads para processar chunks e arquivos de áudio concorrente, reduzindo o tempo total de transcrição.
- **Suporte a Diversos Formatos de Áudio:** Compatível com arquivos `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac` e `.wma`.
- **Acompanhamento de Progresso:** Fornece feedback em tempo real sobre o progresso da transcrição utilizando barras de progresso.
- **Logs Detalhados:** Registro detalhado para monitoramento e depuração.
- **Parâmetros Configuráveis:** Configurações ajustáveis para idioma, número de threads de trabalho e mais.

## Tecnologias Utilizadas

- **Python 3.8+**
- **Biblioteca Transformers** da Hugging Face
- **Librosa:** Para processamento de áudio.
- **Torch (PyTorch):** Backend para computações do modelo.
- **TQDM:** Para barras de progresso.
- **Concurrent Futures:** Para processamento paralelo.
- **Logging:** Para gerenciamento detalhado de logs.

## Pré-requisitos

- **Python 3.8 ou superior** instalado no seu sistema.
- **pip** como gerenciador de pacotes.
- **CUDA** (opcional) para aceleração de GPU, se disponível.

## Instalação

1. **Clone o Repositório:**

   ```bash
   git clone https://github.com/MateusSantosF/pipeline-transcricao-audio.git
   cd pipeline-transcricao-audio
   ```

2. **Crie um Ambiente Virtual (Opcional, mas Recomendado):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

3. **Instale as Dependências Necessárias:**

   ```bash
   pip install -r requirements.txt
   ```

   **Nota:** Se você não possui um `requirements.txt`, pode instalar os pacotes manualmente:

   ```bash
   pip install torch transformers librosa tqdm
   ```

   Para suporte a GPU com PyTorch, siga o [guia oficial de instalação](https://pytorch.org/get-started/locally/) para instalar a versão apropriada.

## Configuração

O script permite configuração através de argumentos de linha de comando. Abaixo estão as opções disponíveis:

- `-i`, `--input_dir`: Caminho para o diretório contendo os arquivos de áudio. **Padrão:** `./audios`
- `-l`, `--language`: Idioma para transcrição (ex: `portuguese`). **Padrão:** `portuguese`
- `-v`, `--verbose`: Aumenta a verbosidade dos logs para nível `DEBUG`.
- `-w`, `--workers`: Número de threads para processamento paralelo. **Padrão:** `8`

## Uso

### Transcrever um Único Arquivo de Áudio

Para transcrever um único arquivo de áudio, você pode especificar o diretório de entrada contendo apenas esse arquivo.

1. **Prepare Seu Arquivo de Áudio:**

   Coloque seu arquivo de áudio (ex: `exemplo.mp3`) no diretório `audios` ou especifique um diretório diferente usando o argumento `--input_dir`.

2. **Execute o Script de Transcrição:**

   ```bash
   python transcribe.py --input_dir ./audios --language portuguese --workers 8 --verbose
   ```

   **Parâmetros:**
   - `--input_dir`: Caminho para o(s) arquivo(s) de áudio.
   - `--language`: Idioma do conteúdo de áudio.
   - `--workers`: Número de threads para processamento.
   - `--verbose`: Habilita logs detalhados.

### Transcrever Múltiplos Arquivos de Áudio em um Diretório

Para transcrever todos os arquivos de áudio suportados em um diretório:

1. **Organize Seus Arquivos de Áudio:**

   Coloque todos os seus arquivos de áudio em um único diretório, por exemplo, `./audios`.

2. **Execute o Script de Transcrição:**

   ```bash
   python transcribe.py --input_dir ./audios --language portuguese --workers 8 --verbose
   ```

   O script irá:

   - Escanear o diretório especificado para formatos de áudio suportados.
   - Criar uma pasta `transcriptions` dentro do diretório de entrada.
   - Transcrever cada arquivo de áudio em paralelo.
   - Salvar cada transcrição como um arquivo `.txt` na pasta `transcriptions`.

## Logs

O script utiliza o módulo `logging` embutido do Python para fornecer logs detalhados.

- **Nível Padrão:** `INFO`
- **Modo Verboso:** Quando a flag `--verbose` está ativa, o nível de log é alterado para `DEBUG` para uma saída mais detalhada.

Os logs incluem:

- Status de carregamento do modelo.
- Status de carregamento e processamento dos arquivos de áudio.
- Progresso do processamento dos chunks.
- Erros e exceções encontradas durante o processamento.

## Otimização de Desempenho

### Processamento Paralelo

O script aproveita o `ThreadPoolExecutor` do módulo `concurrent.futures` para processar chunks e arquivos de áudio em paralelo. Isso reduz significativamente o tempo total de transcrição, especialmente ao lidar com múltiplos arquivos grandes.

- **Configuração de Workers:** Ajuste o parâmetro `--workers` de acordo com os núcleos de CPU e recursos disponíveis do seu sistema. Mais trabalhadores podem acelerar o processamento, mas podem causar contenção de recursos.

### Aceleração com GPU

Se uma GPU estiver disponível, o script a utiliza automaticamente para as computações do modelo, configurando `device_map="auto"`. Isso acelera o processo de transcrição, especialmente para modelos grandes como o `whisper-medium`.

### Processamento por Chunks

Os arquivos de áudio são divididos em chunks (padrão: 30 segundos) para permitir o processamento paralelo. Essa divisão permite que múltiplos chunks sejam transcritos simultaneamente, otimizando ainda mais o desempenho.

## Resolução de Problemas

- **Erros ao Carregar o Modelo:**
  - Verifique sua conexão com a internet para garantir que o modelo possa ser baixado.
  - Certifique-se de que há espaço suficiente em disco.

- **Formatos de Áudio Não Suportados:**
  - O script suporta `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac` e `.wma`. Certifique-se de que seus arquivos de áudio estejam em um desses formatos.

- **Problemas de Memória:**
  - Reduza o número de threads de trabalho usando o parâmetro `--workers`.
  - Utilize tamanhos de chunks menores se enfrentar restrições de memória.

- **Erros com CUDA:**
  - Se estiver usando aceleração GPU, verifique se o CUDA está instalado corretamente e é compatível com sua versão do PyTorch.
  - Verifique a disponibilidade da GPU usando `torch.cuda.is_available()`.

## Contribuindo

Contribuições são bem-vindas! Siga os passos abaixo:

1. **Fork o Repositório.**
2. **Crie uma Branch de Funcionalidade:**

   ```bash
   git checkout -b feature/SuaFuncionalidade
   ```

3. **Comite Suas Alterações:**

   ```bash
   git commit -m "Adiciona sua mensagem aqui"
   ```

4. **Envie para a Branch:**

   ```bash
   git push origin feature/SuaFuncionalidade
   ```

## Agradecimentos

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Transformers da Hugging Face](https://github.com/huggingface/transformers)
- [Librosa](https://librosa.org/)
- [TQDM](https://github.com/tqdm/tqdm)
- [PyTorch](https://pytorch.org/)