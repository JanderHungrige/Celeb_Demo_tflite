import os
import sys
import time
import cv2

import tflite_runtime.interpreter as tflite
import pyarmnn as ann
import numpy as np
import json
import concurrent.futures

times = []


class Ai:
    def __init__(self, model_path, embeddings_path, modeltype='quant',
                 runtime='tflite', preferred_backend='npu'):
        self.model_path = model_path
        self.embeddings_path = embeddings_path
        self.modeltype = modeltype
        self.runtime = runtime
        self.preferred_backend = preferred_backend
        self.width = 224
        self.height = 224

    def initialize(self):
        start = time.time()

        if self.runtime == 'tflite':
            self.init_tflite()
        elif self.runtime == 'armnn':
            self.init_armnn()

        print('Create Embeddigns')
        with open(self.embeddings_path, 'r') as f:
            embeddings_data = json.load(f)

        data = embeddings_data['Embedding']
        self.embeddings = [np.array(data[str(i)]) for i in range(len(data))]

        data = embeddings_data['Name']
        self.names = [np.array(data[str(i)]) for i in range(len(data))]

        data = embeddings_data['File']
        self.files = [np.array(data[str(i)]) for i in range(len(data))]

        self.celeb_embeddings = self.split_data_frame(
                                      self.embeddings,
                                      int(np.ceil(len(self.embeddings)/4)))

        print('Initialization done (duration: {})'.format(time.time() - start))

    def run_inference(self, face):
        #Resize face
        print('Resize face')
        if face.shape > (self.width, self.height):
            face = cv2.resize(face, (self.width, self.height),
                              interpolation=cv2.INTER_AREA)
        elif face.shape < (self.width, self.height):
            face = cv2.resize(face, (self.width, self.height),
                              interpolation=cv2.INTER_CUBIC)

        print('Preprocess')
        if self.modeltype is 'quant':
            samples = np.expand_dims(face, axis=0)
            samples = self.preprocess_input(samples,
                                            data_format='channels_last',
                                            version=3).astype('int8')
        else:
            face = face.astype('float32')
            samples = np.expand_dims(face, axis=0)
            samples = self.preprocess_input(samples,
                                            data_format='channels_last',
                                            version=2)

        assert self.runtime in {'tflite', 'armnn'}

        if self.runtime == 'tflite':
            output_data = self.run_tflite(samples)
        elif self.runtime == 'armnn':
            output_data = self.run_armnn(samples)
        else:
            raise TypeError('Not a valid platform -> {}'.format(self.runtime))

        #return times[-1]

        print('Create EUdist')
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            result_1 = executor.submit(self.faceembedding, output_data,
                                       np.array(self.celeb_embeddings[0]))
            result_2 = executor.submit(self.faceembedding, output_data,
                                       np.array(self.celeb_embeddings[1]))
            result_3 = executor.submit(self.faceembedding, output_data,
                                       np.array(self.celeb_embeddings[2]))
            result_4 = executor.submit(self.faceembedding, output_data,
                                       np.array(self.celeb_embeddings[3]))

        EUdist = []
        if result_1.done() & result_2.done() & result_3.done() & result_4.done():
            EUdist.extend(result_1.result())
            EUdist.extend(result_2.result())
            EUdist.extend(result_3.result())
            EUdist.extend(result_4.result())

        # idx = np.argpartition(EUdist, 10)
        # idx = idx[:10]

        # for i in idx:
        #    print('EUdist: {:8.3f} {} ({})'.format(EUdist[i],
        #                        self.names[i], self.files[i]))

        # with open('dist.txt', 'w') as f:
        #    for i in range(len(EUdist)):
        #        f.write('EUdist: {:8.3f} {} ({})'.format(
        #            EUdist[i], self.names[i], self.files[i]))
        #        f.write('\n')


        idx = np.argpartition(EUdist, 5)                
        folder_name= self.names[idx[0]]
        file_name = self.files[idx[0]] 
        distance = EUdist[idx[0]]
        print('EUdist duration: {}'.format(time.time() - start))

        return folder_name, file_name, distance, idx

    def init_armnn(self):
        self.parser = ann.ITfLiteParser()
        self.network = self.parser.CreateNetworkFromBinaryFile(self.model_path)
        self.input_binding_info = self.parser.GetNetworkInputBindingInfo(0, 'input_1')

        # Create a runtime object that will perform inference.
        self.options = ann.CreationOptions()
        self.runtime = ann.IRuntime(self.options)
        # Choose preferred backends for execution and optimize the network.
        # Backend choices earlier in the list have higher preference.
        if self.preferred_backend=='cpu':
            preferred_backends = [ann.BackendId('CpuAcc')]
        elif self.preferred_backend=='npu':
            preferred_backends = [ann.BackendId('VsiNpu'),
                                 ann.BackendId('CpuAcc'),
                                 ann.BackendId('CpuRef')]
        else:
            raise TypeError('No preferred Backend defined')

        opt_network, messages = ann.Optimize(self.network,
                                             preferred_backends,
                                             self.runtime.GetDeviceSpec(),
                                             ann.OptimizerOptions())
        # Load the optimized network into the runtime.
        net_id, _ = self.runtime.LoadNetwork(opt_network)

    def run_armnn(self, samples):
        print('Invoke PyArmNN')
        start = time.time()
        input_tensors = ann.make_input_tensors([self.input_binding_info],
                                               [samples])
        output_binding_info = self.parser.GetNetworkOutputBindingInfo(0, 'model/output')
        output_tensors = ann.make_output_tensors([output_binding_info])
        self.runtime.EnqueueWorkload(0, input_tensors, output_tensors)
        output_data = ann.workload_tensors_to_ndarray(output_tensors)

        times.append(time.time() - start)
        print('Runtime done ({})'.format(time.time() - start))
        return output_data

    def init_tflite(self):
        try:
            self.interpreter = tflite.Interpreter(self.model_path)
        except ValueError as e:
            print('Failed to find model file: ' + str(e))
            return

        print('Allocate Tensors')
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def run_tflite(self, samples):
        print('Invoke TFlite')
        start = time.time()
        input_shape = self.input_details[0]['shape']
        self.interpreter.set_tensor(self.input_details[0]['index'], samples)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(
                        self.output_details[0]['index'])
        times.append(time.time() - start)
        print('Interpreter done ({})'.format(time.time() - start))
        return output_data

    def split_data_frame(self, df, chunk_size):
        list_of_df = list()
        number_chunks = len(df) // chunk_size + 1
        for i in range(number_chunks):
            list_of_df.append(df[i*chunk_size:(i+1)*chunk_size])

        return list_of_df

    def preprocess_input(self, x, data_format, version):
        x_temp = np.copy(x)
        assert data_format in {'channels_last', 'channels_first'}

        if version == 1:
            if data_format == 'channels_first':
                x_temp = x_temp[:, ::-1, ...]
                x_temp[:, 0, :, :] -= 93.5940
                x_temp[:, 1, :, :] -= 104.7624
                x_temp[:, 2, :, :] -= 129.1863
            else:
                x_temp = x_temp[..., ::-1]
                x_temp[..., 0] -= 93.5940
                x_temp[..., 1] -= 104.7624
                x_temp[..., 2] -= 129.1863

        elif version == 2:
            if data_format == 'channels_first':
                x_temp = x_temp[:, ::-1, ...]
                x_temp[:, 0, :, :] -= 91.4953
                x_temp[:, 1, :, :] -= 103.8827
                x_temp[:, 2, :, :] -= 131.0912
            else:
                x_temp = x_temp[..., ::-1]
                x_temp[..., 0] -= 91.4953
                x_temp[..., 1] -= 103.8827
                x_temp[..., 2] -= 131.0912

        elif version == 3:
            if data_format == 'channels_first':
                x_temp = x_temp[:, ::-1, ...]
                x_temp[:, 0, :, :] -= np.round(91.4953).astype('uint8')
                x_temp[:, 1, :, :] -= np.round(103.8827).astype('uint8')
                x_temp[:, 2, :, :] -= np.round(131.0912).astype('uint8')
            else:
                x_temp = x_temp[..., ::-1]
                x_temp[..., 0] -= np.round(91.4953).astype('uint8')
                x_temp[..., 1] -= np.round(103.8827).astype('uint8')
                x_temp[..., 2] -= np.round(131.0912).astype('uint8')
        else:
            raise NotImplementedError

        return x_temp

    def faceembedding(self, face, celebdata):
        dist = []
        for i in range(len(celebdata)):
            celebs = np.array(celebdata[i])
            dist.append(np.linalg.norm(face - celebs))

        return dist


if __name__ == '__main__':
    #model_file = 'models/tflite/tf220_all_int8.tflite'
    model_file = 'models/tflite/quantized_modelh5-15.tflite'
    #embeddings_file = 'EMBEDDINGS_tf220_all_int8.json'
    embeddings_file = 'EMBEDDINGS_quantized_modelh5-15.json'
    new_ai = Ai(os.path.join(sys.path[0], model_file),
                os.path.join(sys.path[0], embeddings_file),
                modeltype = 'normal', runtime='tflite', preferred_backend='npu')

    new_ai.initialize()
    me = cv2.imread('test_x476_y204_w372_h372.png')
    danny = cv2.imread('danny.jpg')
    fairuza = cv2.imread('fairuza.jpg')
    richard = cv2.imread('richard.jpg')
    shirley = cv2.imread('shirley.jpg')
    vin = cv2.imread('vin.jpg')

    for i in range(100):
        try:
            result = new_ai.run_inference(danny)
            print('DANNY:', result)
            result = new_ai.run_inference(fairuza)
            print('FAIRUZA:', result)
            result = new_ai.run_inference(richard)
            print('RICHARD:', result)
            result = new_ai.run_inference(shirley)
            print('SHIRLEY:', result)
            result = new_ai.run_inference(vin)
            print('VIN:', result)

        except KeyboardInterrupt:
            print('Interrupted')
            break

    with open('times.txt', 'w') as f:
        for t in times:
            f.write(str(t))
            f.write('\n')
