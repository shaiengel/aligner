import torch
from align.handler.transcribe import (audio_to_text, 
                                      audio_to_text_aligner,
                                      audio_to_text_ivirit, 
                                      audio_to_transcribe_ivrit, 
                                      audio_to_transcribe_ivrit_hf,
                                      audio_to_transcribe_fast,
                                      get_model,
                                      convert_audio,
                                      write_to_srt,
                                      clean_alignment_response
                                      )
from align.handler.aligner import aligner
from align.handler.align_mgr import align_repo
from align.services.logger import get_logger
from align.services.docx_util import (read_docx,
                                      remove_marks_for_aligner)
from align.services.statistics import add_probabilties_to_srt, weighted_run_score

logger = get_logger()

logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"CUDA version: {torch.version.cuda}")
logger.info(f"cuDNN : {torch.backends.cudnn.enabled}")

repos = [
    {'audio_repo': 'repo_audio\\brachot', 'audio_file_template': 'Bsafa_Brura-01_BR-{}.mp3', 'doc_repo': 'repo_doc\\Berakhot', 'doc_file_template': 'Berakhot_{}.docx', 'output_repo': 'output_repo', 'start_page': 2}
]


def main():
    
    logger.info("Starting Message Service application")
    #align_repo(repos)
    aligner("repo_audio\\psachim\\Bsafa_Brura-04_PS-111.mp3", ["", "repo_doc\\Pesachim\\Pesachim_111.docx", ""], "output_repo\\psachim")
    
    
    

if __name__ == '__main__':    
    main()

