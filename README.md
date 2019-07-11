# SR-RNNs
The code for the ICML 2019 paper State-Regularized Recurrent Neural Networks (http://proceedings.mlr.press/v97/wang19j.html)

<img src="https://github.com/deepsemantic/sr-rnns/blob/master/SR-RNNs.png" algin="middle"  width="600" >

# Usage and Examples  
   ## 1. SR-GRU on Tomita Grammar
   
   a. To train model, please run: THEANO_FLAGS=mode=FAST_RUN,device=cuda3,floatX=float32 python SR_GRU.py  --tomita_grammar=n. The n is the grammar types (n=1,2,3,4,7) (You can also run SR_GRU_temperature.py, it is a similar implementation but the temperature parameter is considered.)
   
   b. To extract DFA with pre-trained model: THEANO_FLAGS=mode=FAST_RUN,device=cuda3,floatX=float32 python DFA_Extractor.py --tomita_grammar=n
   
   c. In "DFAs" folder, you can find the extracted DFA.
   
   We used the code from [1] to generate train and valid dataset.
   
   ## 2. SR-LSTM-P on Balanced Parenthess 
   
   a. Use "BP_Generator.py" to generated train and valid dataset
   
   b. To train model, please run: THEANO_FLAGS=mode=FAST_RUN,device=cuda3,floatX=float32 python SR_LSTM_P.py
   
   c. The trained model will be saved in "models" folder
   
   d. Run "SR_LSTM_P_inference.py" at inference stage, it calls "plot_transition.py" to plot the state transitions

[1] Weiss, Gail, Yoav Goldberg, and Eran Yahav. "Extracting automata from recurrent neural networks using queries and counterexamples." arXiv preprint arXiv:1711.09576 (2017).      

# Citation
Please cite in your publications if it helps your research:  
        @InProceedings{pmlr-v97-wang19j,  
          title = 	 {State-Regularized Recurrent Neural Networks},  
          author = 	 {Wang, Cheng and Niepert, Mathias},  
          booktitle = 	 {Proceedings of the 36th International Conference on Machine Learning},  
          pages = 	 {6596--6606},  
          year = 	 {2019},  
          editor = 	 {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},  
          volume = 	 {97},  
          series = 	 {Proceedings of Machine Learning Research},  
          address = 	 {Long Beach, California, USA},  
          month = 	 {09--15 Jun},  
          publisher = 	 {PMLR}  
        }
