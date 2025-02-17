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


/******* define what you want to do *******/
#define drop
// #define conv
// #define norm
// #define norm_l2

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
    int quant_2d_list[6][40] = { {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, }, {25, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, }, {13, 25, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, }, {13, 19, 25, 35, 38, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, }, {6, 13, 16, 19, 23, 25, 26, 28, 35, 38, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, }, {1, 3, 4, 6, 9, 13, 15, 16, 19, 20, 23, 25, 26, 28, 30, 31, 33, 35, 36, 38, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, } };
    //int quant_list[40] = {-1};
    
    // for counting the norm of each filter in conv2
    double norm_l1[40] = {0};


    int bit_length;
    int in_num;
    int step_len;
    int step_start;
    int select_num;

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
    select_num = atoi(argv[5]);
    
    int *quant_list = &quant_2d_list[select_num][0];
    //int quant_list[40] = {-1};

    /***** initialize quant list ******/
    // for (int i = 0; i < 40; ++i) quant_list[i] = -1;

    // /***** initialize quant list by step ******/
    // for (int i = 0; i < step_len; ++i) {
    //     quant_list[i] = (step_start + i);
    //     //printf("%d ",(step_start + i));
    // }
    //printf("\n");

    /******** show quant list*********/
    for (int i = 0; i < 40; ++i) printf("%d, ",quant_list[i]);




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


    int class_list[10] = {0};
    int wrong_list[10] = {0};

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
            //input[index] = ((atof(token) / 255));
            data_input_raw[index] = _Q15_i(input[index]);
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
        
        //relu(bufferA, bufferB); swap_buffer(bufferA, bufferB);

        
        //dump_buf(bufferA);
        quant(bufferA, bit_length);
        //dump_buf(bufferA);
        maxpool(bufferA, bufferB, 2); swap_buffer(bufferA, bufferB);
    
        quant(bufferA, bit_length);

    #ifdef quant
        conv2d_fir_quant(bufferA, &conv2_w, &conv2_b, bufferB, quant_list); swap_buffer(bufferA, bufferB);
    
    #endif

    #ifdef norm
        conv2d_fir_quant_norm(bufferA, &conv2_w, &conv2_b, bufferB, norm_l1, in_num); swap_buffer(bufferA, bufferB);
    #endif

    #ifdef norm_l2
        conv2d_fir_quant_norm_l2(bufferA, &conv2_w, &conv2_b, bufferB, norm_l1, in_num); swap_buffer(bufferA, bufferB);
    #endif
        
    #ifdef conv
        conv2d_fir(bufferA, &conv2_w, &conv2_b, bufferB); swap_buffer(bufferA, bufferB);
    #endif

    #ifdef drop
        conv2d_fir_dis(bufferA, &conv2_w, &conv2_b, bufferB, quant_list); swap_buffer(bufferA, bufferB);
    #endif
        //relu(bufferA, bufferB); swap_buffer(bufferA, bufferB);

        // dump_buf(bufferA);
        // quant(bufferA, bit_length);
        quant(bufferA, bit_length);
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
            //printf("%d: %d\n", k, bufferB->data[k]);
            if(bufferB->data[k] > bufferB->data[prediction]) prediction = k;
        }
        //printf("%dth prediction: %d, and = %d\n",i ,prediction, ans);
        
        //
        if (prediction == ans) right++;

        // class_list[ans]++;
        // if (prediction == ans) {
        //     // right ans
        //     right++;  
        // } else {
        //     //wrong ans
        //     wrong_list[ans]++;

        // }
        
    }
    //printf("right cnt = %d\n",right);
    //printf("accuracy = %f\n", ((float)right)/in_num);
    for (int t = 0; t < step_len; ++t) printf("%f\n", (((float)right)/in_num )* 100) ;
    
    // for (int i = 0; i < 10; ++i) {
    //     printf("%d class's wrong prob: %f\n", i,(((float)wrong_list[i])/(float)class_list[i] )* 100);
    // }

#if defined(norm) || defined(norm_l2)
/********* print norm l1 ***********/
    for (int i = 0; i < 40; ++i) {
        //printf("Norm l1_list[%d] = %lf\n",i, norm_l1[i]);
        printf("%lf\n",norm_l1[i]);
    }
    for (int i = 0; i < 40; ++i) {
       // printf("Norm l1_list[%d] = %lf\n",i, norm_l1[i]);
       printf("%lf,",norm_l1[i]);
    }
#endif




    fclose(fp);
    

        

    return 0;
}