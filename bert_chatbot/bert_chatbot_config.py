BERT_CONFIG={
    'bert_model':'./bert_chatbot', 
    'do_lower_case':True, 
    'do_predict':True, 
    'do_train':True, 
    'doc_stride':128, 
    'fp16':False, 
    'gradient_accumulation_steps':1, 
    'learning_rate':3e-05, 
    'local_rank':-1, 
    'loss_scale':128, 
    'max_answer_length':30, 
    'max_query_length':64, 
    'max_seq_length':384, 
    'n_best_size':20, 
    'no_cuda':False, 
    'num_train_epochs':2.0, 
    'optimize_on_cpu':False, 
    'output_dir':'./output_chatbot', 
    'predict_batch_size':8, 
    'predict_file':'./bert_chatbot/kor_dev.json', 
    'train_batch_size':4, 
    'train_file':'./bert_chatbot/train-v1.1.json', 
    'verbose_logging':False, 
    'warmup_proportion':0.1,
    'seed':0
}