#include "nn.h"



Buffer *conv2d_fir(const Buffer *input, const Tensor *weight, const Tensor *bias, Buffer *output) {
    output->ndim = input->ndim;
    output->dims[0] = weight->dims[0];
    output->dims[1] = input->dims[1] - (weight->dims[2] - 1);
    output->dims[2] = input->dims[2] - (weight->dims[3] - 1);

    const unsigned input_channel = input->dims[0];
    const unsigned input_map_size = input->dims[1] * input->dims[2];
    const unsigned filter_size = weight->dims[2] * weight->dims[3];
    const unsigned output_channel = output->dims[0];
    const unsigned output_map_size = output->dims[1] * output->dims[2];

#ifdef CPU
    for(unsigned out_ch = 0; out_ch < output_channel; ++out_ch){ 
        for(unsigned out_i = 0; out_i < output_map_size; ++out_i){

            _iq31 acc = 0;
            const unsigned out_row = (out_i ) / output->dims[2];
            const unsigned out_col = (out_i ) % output->dims[1];


            for(unsigned in_ch = 0; in_ch < input_channel; ++in_ch){
//                if(out_i + out_ch + in_ch == 0) msp_benchmarkStart(MSP_BENCHMARK_BASE, 64);
                /* todo: speedup*/
                for(unsigned row = 0; row < weight->dims[2]; ++row){
                    for(unsigned col = 0; col < weight->dims[3]; ++col){
//                          acc += (input->data[in_ch * input_map_size + (out_row + row) * input->dims[2] + (out_col + col)]) * (weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col]);

                        acc = __saturated_add_iq31(acc, __q15mpyl(input->data[in_ch * input_map_size + (out_row + row) * input->dims[2] + (out_col + col)], ((weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col]& 0XFFFFFF00))));
                            //acc = __saturated_add_iq31(acc, __q15mpyl(input->data[in_ch * input_map_size + (out_row + row) * input->dims[2] + (out_col + col)], ((weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col]))));

                        
                        //printf("%d  %d   %d\n",(int32_t)(weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col]),weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col],(int32_t)( weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col] & 0XFFFFF000));
                    }
                }
//                if(out_i + out_ch + in_ch == 0) printf("timer CONV2D: %lu * %d * %d * %d * %d\n", msp_benchmarkStop(MSP_BENCHMARK_BASE), input_channel, output_channel, output->dims[1], output->dims[2]);
            }

            output->data[out_ch * output_map_size + out_i] = acc >> 16; // 這裡會有問題嗎 ? 好像不會
            //printf("acc = %d, acc shift = %d, output = %d, type output = %d\n",acc, acc >> 16, output->data[out_ch * output_map_size + out_i],(int32_t)output->data[out_ch * output_map_size + out_i]);
           // output->data[out_ch * output_map_size + out_i] = acc;
            output->data[out_ch * output_map_size + out_i] =__saturated_add_q15(output->data[out_ch * output_map_size + out_i], bias->data[out_ch] & 0XFF00);
            //output->data[out_ch * output_map_size + out_i] =__saturated_add_q15(output->data[out_ch * output_map_size + out_i], bias->data[out_ch]);
            output->data[out_ch * output_map_size + out_i] >>= 2;

        }
    }
#endif

}


Buffer *conv2d_fir_quant(const Buffer *input, const Tensor *weight, const Tensor *bias, Buffer *output, int *quant_list) {
    output->ndim = input->ndim;
    output->dims[0] = weight->dims[0];
    output->dims[1] = input->dims[1] - (weight->dims[2] - 1);
    output->dims[2] = input->dims[2] - (weight->dims[3] - 1);

    const unsigned input_channel = input->dims[0];
    const unsigned input_map_size = input->dims[1] * input->dims[2];
    const unsigned filter_size = weight->dims[2] * weight->dims[3];
    const unsigned output_channel = output->dims[0];
    const unsigned output_map_size = output->dims[1] * output->dims[2];
    int quant_index = 0;

#ifdef CPU
    for(unsigned out_ch = 0; out_ch < output_channel; ++out_ch){  //最外這圈在換 kernel
        if (out_ch == quant_list[quant_index]) {
            //quant
            quant_index++;
            for(unsigned out_i = 0; out_i < output_map_size; ++out_i){

                _iq31 acc = 0;

                const unsigned out_row = (out_i ) / output->dims[2];
                const unsigned out_col = (out_i ) % output->dims[1];


                for(unsigned in_ch = 0; in_ch < input_channel; ++in_ch){
    //                if(out_i + out_ch + in_ch == 0) msp_benchmarkStart(MSP_BENCHMARK_BASE, 64);
                    for(unsigned row = 0; row < weight->dims[2]; ++row){
                        for(unsigned col = 0; col < weight->dims[3]; ++col){   
                            // quant
                            acc = __saturated_add_iq31(acc, __q15mpyl((input->data[in_ch * input_map_size + (out_row + row) * input->dims[2] + (out_col + col)] & 0XFFFFF000), ((weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col] & 0XFFFFF000)))); // 因為這個 macro 會先 extend 成 36bit, 所以後面的 & 也要改成 32bit
                        }
                    }
                }
                output->data[out_ch * output_map_size + out_i] = acc >> 16; 
                // quant
                output->data[out_ch * output_map_size + out_i] =__saturated_add_q15(output->data[out_ch * output_map_size + out_i], bias->data[out_ch] & 0XF000); 
                output->data[out_ch * output_map_size + out_i] >>= 2;

            }
          
        } else {
            //no quant
            for(unsigned out_i = 0; out_i < output_map_size; ++out_i){
                _iq31 acc = 0;
                const unsigned out_row = (out_i ) / output->dims[2];
                const unsigned out_col = (out_i ) % output->dims[1];


                for(unsigned in_ch = 0; in_ch < input_channel; ++in_ch){
    //                if(out_i + out_ch + in_ch == 0) msp_benchmarkStart(MSP_BENCHMARK_BASE, 64);
                    for(unsigned row = 0; row < weight->dims[2]; ++row){
                        for(unsigned col = 0; col < weight->dims[3]; ++col){
                            // no quant
                            acc = __saturated_add_iq31(acc, __q15mpyl(input->data[in_ch * input_map_size + (out_row + row) * input->dims[2] + (out_col + col)], ((weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col] & 0XFFFFFF00))));
                        }
                    }                
                }

                output->data[out_ch * output_map_size + out_i] = acc >> 16; 
                output->data[out_ch * output_map_size + out_i] =__saturated_add_q15(output->data[out_ch * output_map_size + out_i], bias->data[out_ch] & 0XFF00);
                output->data[out_ch * output_map_size + out_i] >>= 2;
            }
        }
       

//
    }
#endif

}


Buffer *conv2d_fir_quant_norm(const Buffer *input, const Tensor *weight, const Tensor *bias, Buffer *output) {
    output->ndim = input->ndim;
    output->dims[0] = weight->dims[0];
    output->dims[1] = input->dims[1] - (weight->dims[2] - 1);
    output->dims[2] = input->dims[2] - (weight->dims[3] - 1);

    const unsigned input_channel = input->dims[0];
    const unsigned input_map_size = input->dims[1] * input->dims[2];
    const unsigned filter_size = weight->dims[2] * weight->dims[3];
    const unsigned output_channel = output->dims[0];
    const unsigned output_map_size = output->dims[1] * output->dims[2];
    

#ifdef CPU
    for(unsigned out_ch = 0; out_ch < output_channel; ++out_ch){  //最外這圈在換 kernel
    long long Norm = 0;
        for(unsigned out_i = 0; out_i < output_map_size; ++out_i){

            _iq31 acc1 = 0;
            _iq31 acc2 = 0;
            fixed quant_output = 0;
           // int64_t  acc3 = 0;
            const unsigned out_row = (out_i ) / output->dims[2];
            const unsigned out_col = (out_i ) % output->dims[1];


            for(unsigned in_ch = 0; in_ch < input_channel; ++in_ch){
//                if(out_i + out_ch + in_ch == 0) msp_benchmarkStart(MSP_BENCHMARK_BASE, 64);
                /* todo: speedup*/
                for(unsigned row = 0; row < weight->dims[2]; ++row){
                    for(unsigned col = 0; col < weight->dims[3]; ++col){
//                          acc += (input->data[in_ch * input_map_size + (out_row + row) * input->dims[2] + (out_col + col)]) * (weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col]);
                        //if(out_ch <= 100) {
                            // no quant
                            acc1 = __saturated_add_iq31(acc1, __q15mpyl(input->data[in_ch * input_map_size + (out_row + row) * input->dims[2] + (out_col + col)], ((weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col] & 0XFFFFFF00))));
                       // } else {
                            // quant
                            acc2 = __saturated_add_iq31(acc2, __q15mpyl((input->data[in_ch * input_map_size + (out_row + row) * input->dims[2] + (out_col + col)] & 0XFFFFF000), ((weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col] & 0XFFFFF000)))); // 因為這個 macro 會先 extend 成 36bit, 所以後面的 & 也要改成 32bit
                           // acc3 = __saturated_add_iq64(acc3, __q15mpyl((input->data[in_ch * input_map_size + (out_row + row) * input->dims[2] + (out_col + col)] & 0XFFFFF000), ((weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col] & 0XFFFFF000))));
                       // }
                            //acc = __saturated_add_iq31(acc, __q15mpyl(input->data[in_ch * input_map_size + (out_row + row) * input->dims[2] + (out_col + col)], ((weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col] & 0XFFFFFF00))));
                            //acc = __saturated_add_iq31(acc, __q15mpyl(input->data[in_ch * input_map_size + (out_row + row) * input->dims[2] + (out_col + col)], ((weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col]))));

                              
                            // printf("unquant input = %d, weight = %d\n",input->data[in_ch * input_map_size + (out_row + row) * input->dims[2] + (out_col + col)], (weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col] & 0XFFFFFF00));
                            // printf("quant input = %d, weight = %d\n",input->data[in_ch * input_map_size + (out_row + row) * input->dims[2] + (out_col + col)] & 0XFFFFF000, (weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col] & 0XFFFFF000));
                            // printf("acc1 = %ld, acc2 =%ld, acc3 = %lld\n\n", acc1, acc2, acc3);  
                        
                        
                        //printf("%d  %d   %d\n",(int32_t)(weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col]),weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col],(int32_t)( weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col] & 0XFFFFF000));
                    }
                }
//                if(out_i + out_ch + in_ch == 0) printf("timer CONV2D: %lu * %d * %d * %d * %d\n", msp_benchmarkStop(MSP_BENCHMARK_BASE), input_channel, output_channel, output->dims[1], output->dims[2]);
            }
            //printf("--------------------------------------------------\n\n\n");

            output->data[out_ch * output_map_size + out_i] = acc1 >> 16; 
            quant_output = acc2 >>16;
            //printf("acc = %d, acc shift = %d, output = %d, type output = %d\n",acc, acc >> 16, output->data[out_ch * output_map_size + out_i],(int32_t)output->data[out_ch * output_map_size + out_i]);
           // output->data[out_ch * output_map_size + out_i] = acc;
           //if(out_ch <= 100) {
            // no quant
                output->data[out_ch * output_map_size + out_i] =__saturated_add_q15(output->data[out_ch * output_map_size + out_i], bias->data[out_ch] & 0XFF00);
           //} else {
            // quant
            //    output->data[out_ch * output_map_size + out_i] =__saturated_add_q15(output->data[out_ch * output_map_size + out_i], bias->data[out_ch] & 0XF000);
                quant_output =__saturated_add_q15(quant_output, bias->data[out_ch] & 0XF000);
           //}
            
            output->data[out_ch * output_map_size + out_i] >>= 2;
            quant_output >>= 2;

            //printf("filter 0 output, 8bit :%d, 4bit: %d\n",output->data[out_ch * output_map_size + out_i], quant_output);
            long long int diff = output->data[out_ch * output_map_size + out_i] - quant_output;
            //Norm += (unsigned long long)((int32_t)(output->data[out_ch * output_map_size + out_i] - (int32_t)quant_output) * ((int32_t)output->data[out_ch * output_map_size + out_i] - (int32_t)quant_output));
            Norm += (diff * diff);
            //printf("diff = %lld norm now = %lld\n",diff ,Norm);
        

        }
        if (out_ch <= 40) {
            printf("Norm of filter %d = %lld\n",out_ch, Norm);
        }

    }
#endif

}


Buffer *conv2d_fir_dis(const Buffer *input, const Tensor *weight, const Tensor *bias, Buffer *output, int *quant_list) {
    output->ndim = input->ndim;
    output->dims[0] = weight->dims[0];
    output->dims[1] = input->dims[1] - (weight->dims[2] - 1);
    output->dims[2] = input->dims[2] - (weight->dims[3] - 1);

    const unsigned input_channel = input->dims[0];
    const unsigned input_map_size = input->dims[1] * input->dims[2];
    const unsigned filter_size = weight->dims[2] * weight->dims[3];
    const unsigned output_channel = output->dims[0];
    const unsigned output_map_size = output->dims[1] * output->dims[2];
    int quant_index = 0;

#ifdef CPU
    for(unsigned out_ch = 0; out_ch < output_channel; ++out_ch){  //最外這圈在換 kernel
        if (out_ch == quant_list[quant_index]) {
            //quant
            quant_index++;
            for(unsigned out_i = 0; out_i < output_map_size; ++out_i){

                _iq31 acc = 0;

                const unsigned out_row = (out_i ) / output->dims[2];
                const unsigned out_col = (out_i ) % output->dims[1];


                for(unsigned in_ch = 0; in_ch < input_channel; ++in_ch){
    //                if(out_i + out_ch + in_ch == 0) msp_benchmarkStart(MSP_BENCHMARK_BASE, 64);
                    for(unsigned row = 0; row < weight->dims[2]; ++row){
                        for(unsigned col = 0; col < weight->dims[3]; ++col){   
                            // discard
                            acc = 0;
                        }
                    }
                }
                output->data[out_ch * output_map_size + out_i] = acc >> 16; 
                // quant
                //output->data[out_ch * output_map_size + out_i] =__saturated_add_q15(output->data[out_ch * output_map_size + out_i], bias->data[out_ch] & 0XF000); 
                output->data[out_ch * output_map_size + out_i] >>= 2;

            }
          
        } else {
            //no quant
            for(unsigned out_i = 0; out_i < output_map_size; ++out_i){
                _iq31 acc = 0;
                const unsigned out_row = (out_i ) / output->dims[2];
                const unsigned out_col = (out_i ) % output->dims[1];


                for(unsigned in_ch = 0; in_ch < input_channel; ++in_ch){
    //                if(out_i + out_ch + in_ch == 0) msp_benchmarkStart(MSP_BENCHMARK_BASE, 64);
                    for(unsigned row = 0; row < weight->dims[2]; ++row){
                        for(unsigned col = 0; col < weight->dims[3]; ++col){
                            // no quant
                            acc = __saturated_add_iq31(acc, __q15mpyl(input->data[in_ch * input_map_size + (out_row + row) * input->dims[2] + (out_col + col)], ((weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col] & 0XFFFFFF00))));
                        }
                    }                
                }

                output->data[out_ch * output_map_size + out_i] = acc >> 16; 
                output->data[out_ch * output_map_size + out_i] =__saturated_add_q15(output->data[out_ch * output_map_size + out_i], bias->data[out_ch] & 0XFF00);
                output->data[out_ch * output_map_size + out_i] >>= 2;
            }
        }
       

//
    }
#endif

}
