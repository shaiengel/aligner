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
    #audio_to_transcribe_fast("rega.mp3")
    #result = audio_to_transcribe_fast("babakama5.mp3")
    
    #text = read_docx("repo_doc\\16.docx")
    #text = remove_marks_for_aligner(text)
    #model = get_model()   
    #audio = convert_audio("repo_doc\\Bsafa_Brura-26_SV-16.mp3")
    #audio = "repo_doc\\Bsafa_Brura-26_SV-16.mp3"
    #result, _ = audio_to_text_aligner(model, audio, text, "output")
    #result = clean_alignment_response(result)
    #write_to_srt(result, "repo_doc\\Bsafa_Brura-26_SV-16.mp3", "output")
    #result_probability_list = add_probabilties_to_srt("output\\Bsafa_Brura-26_SV-16.srt", result.ori_dict["segments"][0]["words"]) 
    #w = weighted_run_score(result_probability_list)
    #print(f"weighted_run_score: w = {w}")
   
    #aligner("repo\\brachot_audio\\Bsafa_Brura-01_BR-64.mp3", ["repo\\Berakhot\\Berakhot_63.docx", "repo\\Berakhot\\Berakhot_64.docx", ""])
    #aligner("repo\\Bsafa_Brura-26_SV-16.mp3", ["repo\\Bsafa_Brura-26_SV-15.docx", "repo\\Bsafa_Brura-26_SV-16.docx", "repo\\Bsafa_Brura-26_SV-17.docx"])
    align_repo(repos)
    #aligner("repo_doc\\Bsafa_Brura-26_SV-16.mp3", ["repo_doc\\Shevuot\\Shevuot_15.docx", "repo_doc\\Shevuot\\Shevuot_16.docx", "repo_doc\\Shevuot\\Shevuot_17.docx"])
    
    #text0 = read_docx("rega.docx")
    #clean_text0 = remove_marks_for_aligner(text0)
    #output_file, response = audio_to_text("rega.mp3", clean_text0)
    #add_probabilties_to_srt(output_file, response.ori_dict["segments"][0]["words"]) 
    

if __name__ == '__main__':    
    main()

