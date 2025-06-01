import torch
from align.handler.transcribe import (audio_to_text, 
                                      audio_to_text_ivirit, 
                                      audio_to_transcribe_ivrit, 
                                      audio_to_transcribe_ivrit_hf,
                                      audio_to_transcribe_fast
                                      )
from align.handler.aligner import aligner
from align.services.logger import init_logger, format_rtl
from align.services.docx_util import (read_docx,
                                      remove_marks_for_aligner)
from align.services.statistics import add_probabilties_to_srt

logger = init_logger(__name__)

logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"CUDA version: {torch.version.cuda}")
logger.info(f"cuDNN : {torch.backends.cudnn.enabled}")


def main():
    
    logger.info("Starting Message Service application")
    #audio_to_transcribe_fast("rega.mp3")
    #result = audio_to_transcribe_fast("babakama5.mp3")
    #text = clean_text_from_docx("Bsafa_Brura-26_SV-3.docx")
    #save_cleaned_text_to_docx(text, "Bsafa_Brura-26_SV-3_fix.docx")
    #combined = combine_docx_content("repo\\Bsafa_Brura-26_SV-15.docx", "repo\\Bsafa_Brura-26_SV-16.docx", -9)
    #text = remove_marks(combined)
    #logger.info(format_rtl(text))
    #audio_to_text("repo\\Bsafa_Brura-26_SV-16.mp3", text)
    #search_starting("repo\\Bsafa_Brura-26_SV-3.mp3", ["repo\\Bsafa_Brura-26_SV-2.docx", "repo\\Bsafa_Brura-26_SV-3.docx"])
    #aligner("repo\\Bsafa_Brura-26_SV-22.mp3", ["repo\\Bsafa_Brura-26_SV-21.docx", "repo\\Bsafa_Brura-26_SV-22.docx", "repo\\Bsafa_Brura-26_SV-23.docx"])
    
    text0 = read_docx("rega.docx")
    clean_text0 = remove_marks_for_aligner(text0)
    output_file, response = audio_to_text("rega.mp3", clean_text0)
    add_probabilties_to_srt(output_file, response.ori_dict["segments"][0]["words"]) 
    

if __name__ == '__main__':    
    main()

