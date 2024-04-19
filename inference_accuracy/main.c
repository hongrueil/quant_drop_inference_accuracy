#include <stdlib.h>
#include "inference/nn.h"
/************* input **************/
/*
input data is from: https://github.com/pjreddie/mnist-csv-png/blob/master/process_mnist.py  
which has 10000 test data from mnist

*/



/********** include model *************/
//#include "dataset/mnist/input.h"
#include "model/hawaii/model.h"
//#define in_num 5000
#define mean 0.1307
#define sd 0.3081


Buffer data_input = {
    .ndim = 3,
    .dims = {1, 28, 28},
    .bw = 0,
    .data = NULL
    //.data = (fixed *)data_input_raw[0]
};


/********** buffer A **********/
fixed bufferA_data[11600] = {0};
Buffer bufferA_tensor = {
    .ndim = 0,
    .dims = {0},
    .bw = 0,
    .data = (fixed *)bufferA_data
};

/********** buffer B **********/
fixed bufferB_data[11600] = {0};
Buffer bufferB_tensor = {
    .ndim = 0,
    .dims = {0},
    .bw = 0,
    .data = (fixed *)bufferB_data
};


int main(int argc, char *argv[]) {
    int right = 0;

    Buffer *bufferA, *bufferB;
    bufferA = &bufferA_tensor;
    bufferB = &bufferB_tensor;

    // for select quantization channel
    int quant_list[40] = {-1};
    //int quant_list[40] = {13, 39 -1};


    int bit_length;
    int in_num;
    int step_len;
    int step_start;

//    int filter_num;
    //printf("input bit length: ");
    //scanf("%d",&bit_length);
    bit_length = atoi(argv[1]);
    //printf("in num: ");
    //scanf("%d",&in_num);
    in_num = atoi(argv[2]);
    //printf("step_len: ");
   // scanf("%d",&step_len);
    step_len = atoi(argv[3]);
    //printf("step_start: ");
    //scanf("%d",&step_start);
    step_start = atoi(argv[4]);
    //printf("%d, %d , %d, %d\n",bit_length,in_num,step_len,step_start);
    //printf("Step = %d, Start = %d\n",step_len, step_start);


    for (int i = 0; i < 40; ++i) quant_list[i] = -1;

    for (int i = 0; i < step_len; ++i) {
        quant_list[i] = (step_start + i);
        //printf("%d ",(step_start + i));
    }
    //printf("\n");





    FILE *fp = fopen("./dataset/input/mnist_test.csv", "r");
    if (fp == NULL) {
        fprintf(stderr, "fopen() failed.\n");
        exit(EXIT_FAILURE);
    }
    //printf("222time = %d\n", time);
    char str[2400];
    char *token;
    float input[28*28];
    fixed data_input_raw[28 * 28];


    

    for (int i = 0; i < in_num; ++i) {
        //printf("\nresult of %d input\n",in_num);
    //
        int ans;

    // load input from csv

        int index = 0;
        fgets(str, 2400, fp);
        token = strtok(str, ","); 
        ans = atoi(token); //the answer of this input
        
        // load csv which spilt by ','
        while (token != NULL) {
            //printf("%s ", token);
            token = strtok(NULL, ",");
            input[index] = (((atof(token) / 255) - mean) / sd);
            data_input_raw[index] = _Q15(input[index]);
            index++;
        }



    //initial
        data_input.data = (fixed *)data_input_raw;




    // load input into buffer A
        bufferA->ndim = data_input.ndim;
        memcpy(bufferA->dims, data_input.dims, 6);
        memcpy(bufferA->data, data_input.data, 28*28*sizeof(_q15));

    // inference
        //int bit_length  = 8;
        quant(bufferA, bit_length);
        conv2d_fir(bufferA, &conv1_w, &conv1_b, bufferB); swap_buffer(bufferA, bufferB);
        
        //dump_buf(bufferA);
        quant(bufferA, bit_length);
        //dump_buf(bufferA);
        maxpool(bufferA, bufferB, 2); swap_buffer(bufferA, bufferB);
    
        quant(bufferA, bit_length);
        conv2d_fir_quant(bufferA, &conv2_w, &conv2_b, bufferB, quant_list); swap_buffer(bufferA, bufferB);
//           conv2d_fir(bufferA, &conv2_w, &conv2_b, bufferB); swap_buffer(bufferA, bufferB);
        //conv2d_fir_dis(bufferA, &conv2_w, &conv2_b, bufferB, quant_list); swap_buffer(bufferA, bufferB);
    // dump_buf(bufferA);
        quant(bufferA, 4);
        //dump_buf(bufferA);
        
        maxpool(bufferA, bufferB, 2); swap_buffer(bufferA, bufferB);
        

        flatten(bufferA);
    
        fc(bufferA, &fc1_w, &fc1_b, bufferB); swap_buffer(bufferA, bufferB);
        quant(bufferA, bit_length);
        relu(bufferA, bufferB); swap_buffer(bufferA, bufferB);
        //quant(bufferA, bit_length);
        fc(bufferA, &fc2_w, &fc2_b, bufferB);

    //prediction
        unsigned prediction = 0;
        for(int k = 0; k < 10; ++k){
            //printf("%d: %d\n", i, bufferB->data[i]);
            if(bufferB->data[k] > bufferB->data[prediction]) prediction = k;
        }
        //printf("%dth prediction: %d, and = %d\n",i ,prediction, ans);
    
        if (prediction == ans) right++;
        
    }
    //printf("right cnt = %d\n",right);
    //printf("accuracy = %f\n", ((float)right)/in_num);
    for (int t = 0; t < step_len; ++t) printf("%f\n", (((float)right)/in_num )* 100) ;
    


    // printf("%d    %d",_Q15(10),  _Q15(10) & bit_mask_12);
    // printf("%d    %d",_Q7(10),  _Q7(10) & bit_mask_4);
    fclose(fp);
    

        
   // }
    return 0;
}