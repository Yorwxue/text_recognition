import tensorflow as tf
from basenet.transformation import TPS_SpatialTransformerNetwork
from basenet.extration import ResNet
from basenet.prediction import Attention


class Model(tf.keras.Model):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.stages = {'Trans': args.Transformation, 'Feat': args.FeatureExtraction,
                       'Seq': args.SequenceModeling, 'Pred': args.Prediction}

        # Transformation
        if self.stages["Trans"] == "TPS":
            self.Transformation = TPS_SpatialTransformerNetwork(args.F, I_r_size=(args.imgH, args.imgW))
        else:
            print('No Transformation module specified')

        # FeatureExtraction
        if self.stages["Feat"] == 'ResNet':
            self.FeatureExtraction = ResNet(args.output_channel)
        else:
            raise NotImplementedError

        # Sequence modeling
        if self.stages["Seq"] == 'BiLSTM':
            self.SequenceModeling = tf.keras.Model([
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(args.hidden_size, return_sequences=True)),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(args.hidden_size, return_sequences=True))
            ])
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        # Prediction
        if self.stages["Pred"] == 'Attn':
            self.Prediction = Attention(args.hidden_size)
        else:
            raise NotImplementedError

    def call(self, input, text, is_train=True):
        # Transformation
        if self.stages["Trans"] == "TPS":
            input = self.Transformation(input)

        # FeatureExtraction
        visual_feature = self.FeatureExtraction(input)
        visual_feature = tf.transpose(visual_feature, (0, 2, 3, 1))  # [b, h, w, c] -> [b, w, c, h]

        # Sequence modeling
        contextual_feature = self.SequenceModeling(visual_feature)

        # Prediction
        prediction = self.Prediction(contextual_feature.contiguous(), text, is_train,
                                     batch_max_length=self.opt.batch_max_length)

        return prediction
