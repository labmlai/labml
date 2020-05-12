from labml.internal.configs.processor_dict import ConfigProcessorDict

processor = ConfigProcessorDict({'cnn_size': 10, 'batch_size': 12}, {'batch_size': 2})
processor()
processor.print()
