import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa
from tqdm import tqdm
import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuração do logger
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def carregar_modelo(model_id="openai/whisper-medium"):
    """
    Carrega o modelo Whisper e o processador da Hugging Face.
    """
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    logging.info(f"Carregando o modelo '{model_id}'...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.generation_config.language = "<|pt|>"
    model.generation_config.task = "transcribe"
    processor = AutoProcessor.from_pretrained(model_id)
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        chunk_length_s=30,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
    )

    logging.info("Modelo carregado com sucesso.")
    return pipe

def process_chunk(pipe, audio_chunk):
    """
    Processa um único chunk de áudio e retorna a transcrição.
    """
    try:
        resultado = pipe(
            audio_chunk,
            return_timestamps=False
        )
        return resultado["text"]
    except Exception as e:
        logging.error(f"Erro ao transcrever o chunk: {e}")
        return ""

def transcrever_audio(pipe, file_path, chunk_length_s=30, max_workers=4) -> str:
    """
    Transcreve o conteúdo de um arquivo de áudio para texto usando o pipeline Whisper, processando em chunks de forma paralela.
    """
    try:
        logging.debug(f"Carregando arquivo de áudio: {file_path}")
        audio, rate = librosa.load(file_path, sr=16000)
    except Exception as e:
        logging.error(f"Erro ao carregar '{file_path}': {e}")
        return ""
    
    total_duration = librosa.get_duration(y=audio, sr=rate)
    num_chunks = int(total_duration // chunk_length_s) + 1
    logging.info(f"Duração total do áudio: {total_duration:.2f} segundos.")
    logging.info(f"Dividindo o áudio em {num_chunks} chunks de {chunk_length_s} segundos cada.")
    transcriptions = ["" for _ in range(num_chunks)]

    # Criação dos chunks
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_length_s
        end = min((i + 1) * chunk_length_s, total_duration)
        audio_chunk = audio[int(start * rate):int(end * rate)]
        chunks.append((i, audio_chunk))
        logging.debug(f"Preparado chunk {i+1}/{num_chunks}: {start:.2f}s a {end:.2f}s.")

    # Processamento paralelo dos chunks
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_chunk, pipe, chunk[1]): chunk[0] for chunk in chunks}
        with tqdm(total=num_chunks, desc="Chunks Transcritos", unit="chunk") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    transcriptions[idx] = future.result()
                    logging.debug(f"Transcrição do chunk {idx+1}/{num_chunks} concluída.")
                except Exception as e:
                    logging.error(f"Erro ao transcrever o chunk {idx+1}: {e}")
                pbar.update(1)
    
    transcription = " ".join(transcriptions)
    return transcription

def transcrever_pasta(audio_directory, language="pt", max_workers=4):
    """
    Transcreve todos os arquivos de áudio em um diretório e salva cada transcrição em um arquivo separado
    dentro da pasta 'transcriptions'. Processa múltiplos arquivos em paralelo.
    """
    pipe = carregar_modelo()

    supported_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma')
    arquivos = [f for f in os.listdir(audio_directory) if f.lower().endswith(supported_extensions)]

    if not arquivos:
        logging.warning(f"Nenhum arquivo de áudio encontrado em '{audio_directory}'.")
        return

    logging.info(f"Encontrados {len(arquivos)} arquivos de áudio. Iniciando a transcrição...")

    # Cria a pasta "transcriptions" dentro do diretório de entrada
    transcriptions_dir = os.path.join(audio_directory, "transcriptions")
    os.makedirs(transcriptions_dir, exist_ok=True)

    # Processamento paralelo dos arquivos de áudio
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(transcrever_audio, pipe, os.path.join(audio_directory, arquivo)): arquivo for arquivo in arquivos}
        with tqdm(total=len(arquivos), desc="Arquivos Transcritos", unit="arquivo") as pbar:
            for future in as_completed(futures):
                arquivo = futures[future]
                try:
                    texto = future.result()
                    output_path = os.path.join(transcriptions_dir, os.path.splitext(arquivo)[0] + ".txt")
                    
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(texto)
                    logging.info(f"Transcrição salva em '{output_path}'.")
                except Exception as e:
                    logging.error(f"Erro ao transcrever '{arquivo}': {e}")
                pbar.update(1)

    logging.info("Transcrição de todos os arquivos concluída.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcrever arquivos de áudio em uma pasta para texto usando Whisper.")
    parser.add_argument(
        "-i", "--input_dir", type=str, default="./audios", help="Caminho para a pasta contendo os arquivos de áudio."
    )
    parser.add_argument(
        "-l", "--language", type=str, default="portuguese", help="Idioma para a transcrição (ex: 'portuguese')."
    )
    parser.add_argument(
        "-v", "--verbose", action='store_true', help="Aumenta a verbosidade do log."
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=8, help="Número de threads para processamento paralelo."
    )

    args = parser.parse_args()

    # Ajusta o nível de log com base no argumento --verbose
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    if not os.path.isdir(args.input_dir):
        logging.error(f"A pasta especificada '{args.input_dir}' não existe.")
        exit(1)

    transcrever_pasta(args.input_dir, language=args.language, max_workers=args.workers)
