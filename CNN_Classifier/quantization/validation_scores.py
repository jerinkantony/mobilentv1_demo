def topn_accuracy(pred_list, act_list, n):	#calculate top n accuracy, n=1,5
    good_count = 0
    for i in range(len(act_list)):
            topn = []
            topn_index = []
            topn = sorted(pred_list[i], reverse=True)[:n]

            for a in topn:
                topn_index.append(pred_list[i].index(a))

            if act_list[i] in topn_index:
                good_count+=1
    
    print('Top',n,':', good_count/len(act_list) * 100, '%')

def mean_avg_precision(preds,truths):		#calculate mean average precision score
    tp_counter=0
    cumulate_precision = 0
    for i in range(len(truths)):
        if preds[i] == truths[i]:
            tp_counter += 1
            cumulate_precision += (float(tp_counter)/float(i+1))
    if tp_counter != 0:
            mAP = cumulate_precision/len(truths)
    print("mean average precision is", mAP)
