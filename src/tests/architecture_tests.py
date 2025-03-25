import numpy as np
import tensorflow as tf

def get_angles_test(target):
    position = 4
    d_model = 16
    pos_m = np.arange(position)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]

    result = target(pos_m, dims, d_model)

    assert type(result) == np.ndarray, "You must return a numpy ndarray"
    assert result.shape == (position, d_model), f"Wrong shape. We expected: ({position}, {d_model})"
    assert np.sum(result[0, :]) == 0
    assert np.isclose(np.sum(result[:, 0]), position * (position - 1) / 2)
    even_cols =  result[:, 0::2]
    odd_cols = result[:,  1::2]
    assert np.all(even_cols == odd_cols), "Submatrices of odd and even columns must be equal"
    limit = (position - 1) / np.power(10000,14.0/16.0)
    assert np.isclose(result[position - 1, d_model -1], limit ), f"Last value must be {limit}"

    print("\033[92mAll tests passed")
    
def positional_encoding_test(target, get_angles):
    position = 8
    d_model = 16

    pos_encoding = target(position, d_model)
    sin_part = pos_encoding[:, :, 0::2]
    cos_part = pos_encoding[:, :, 1::2]

    assert tf.is_tensor(pos_encoding), "Output is not a tensor"
    assert pos_encoding.shape == (1, position, d_model), f"Wrong shape. We expected: (1, {position}, {d_model})"

    ones = sin_part ** 2  +  cos_part ** 2
    assert np.allclose(ones, np.ones((1, position, d_model // 2))), "Sum of square pairs must be 1 = sin(a)**2 + cos(a)**2"
    
    angs = np.arctan(sin_part / cos_part)
    angs[angs < 0] += np.pi
    angs[sin_part.numpy() < 0] += np.pi
    angs = angs % (2 * np.pi)
    
    pos_m = np.arange(position)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]

    trueAngs = get_angles(pos_m, dims, d_model)[:, 0::2] % (2 * np.pi)
    
    assert np.allclose(angs[0], trueAngs), "Did you apply sin and cos to even and odd parts respectively?"
 
    print("\033[92mAll tests passed")
    
def scaled_dot_product_attention_test(target):
    q = np.array([[1, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1]]).astype(np.float32)
    k = np.array([[1, 1, 0, 1], [1, 0, 1, 1 ], [0, 1, 1, 0], [0, 0, 0, 1]]).astype(np.float32)
    v = np.array([[0, 0], [1, 0], [1, 0], [1, 1]]).astype(np.float32)

    attention, weights = target(q, k, v, None)
    assert tf.is_tensor(weights), "Weights must be a tensor"
    assert tuple(tf.shape(weights).numpy()) == (q.shape[0], k.shape[1]), f"Wrong shape. We expected ({q.shape[0]}, {k.shape[1]})"
    assert np.allclose(weights, [[0.2589478,  0.42693272, 0.15705977, 0.15705977],
                                   [0.2772748,  0.2772748,  0.2772748,  0.16817567],
                                   [0.33620113, 0.33620113, 0.12368149, 0.2039163 ]]), "Wrong unmasked weights"

    assert tf.is_tensor(attention), "Output must be a tensor"
    assert tuple(tf.shape(attention).numpy()) == (q.shape[0], v.shape[1]), f"Wrong shape. We expected ({q.shape[0]}, {v.shape[1]})"
    assert np.allclose(attention, [[0.74105227, 0.15705977],
                                   [0.7227253,  0.16817567],
                                   [0.6637989,  0.2039163 ]]), "Wrong unmasked attention"

    mask = np.array([[[1, 1, 0, 1], [1, 1, 0, 1], [1, 1, 0, 1]]])
    attention, weights = target(q, k, v, mask)

    assert np.allclose(weights, [[0.30719590187072754, 0.5064803957939148, 0.0, 0.18632373213768005],
                                 [0.3836517333984375, 0.3836517333984375, 0.0, 0.2326965481042862],
                                 [0.3836517333984375, 0.3836517333984375, 0.0, 0.2326965481042862]]), "Wrong masked weights"
    assert np.allclose(attention, [[0.6928040981292725, 0.18632373213768005],
                                   [0.6163482666015625, 0.2326965481042862], 
                                   [0.6163482666015625, 0.2326965481042862]]), "Wrong masked attention"
    
    print("\033[92mAll tests passed")


def EncoderLayer_test(target):
    q = np.array([[[1, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1]]]).astype(np.float32)
    encoder_layer1 = target(4, 2, 8)
    tf.random.set_seed(10)
    
    # Run encoder layer with training=True
    encoded = encoder_layer1(q, training=True, mask=np.array([[1, 0, 1]]))
    
    assert tf.is_tensor(encoded), "Wrong type. Output must be a tensor"
    assert tuple(tf.shape(encoded).numpy()) == (1, q.shape[1], q.shape[2]), f"Wrong shape. We expected ((1, {q.shape[1]}, {q.shape[2]}))"
    assert np.allclose(encoded.numpy(), 
                       [[ 0.9754241, 0.15179634, 0.52959377, -1.6568142 ],
                        [-0.24592467, 0.4422113, 1.2665371, -1.4628237 ],
                        [ 1.6900116, -0.35137546, -0.4198987, -0.91873735]], atol=1e-5), "Wrong values when training=True"

    # Run encoder layer with training=False
    encoded = encoder_layer1(q, training=False, mask=np.array([[1, 1, 0]]))
    
    assert np.allclose(encoded.numpy(), 
                       [[ 0.73325014, 0.01539412, 0.8870007, -1.6356449 ],
                        [-0.48556724, 0.3304677, 1.4271768, -1.2720773 ],
                        [ 1.6128806, -0.39568534, -0.10796563, -1.1092294 ]], atol=1e-5), "Wrong values when training=False"
    
    print("\033[92mAll tests passed")

def Encoder_test(target):
    tf.random.set_seed(10)
    
    embedding_dim = 4
    
    encoderq = target(num_layers=2,
                      embedding_dim=embedding_dim,
                      num_heads=2,
                      fully_connected_dim=8,
                      input_vocab_size=32,
                      maximum_position_encoding=5)
    
    x = np.array([[2, 1, 3], [1, 2, 0]])
    
    # Case 1: training=True, mask=None
    encoderq_output = encoderq(x, training=True, mask=None)
    expected_output_1 = np.array([
        [[ 0.0870375,  -1.23968,    -0.3700725,   1.5227151 ],
        [ 0.60949683,  0.7264111,   0.38288212, -1.7187902 ],
        [ 1.1098912,  -1.0808123,  -0.90884334,  0.8797645 ]],
        [[ 0.04190131, -1.0852709,  -0.545114,    1.5884836 ],
        [ 1.0634973,  -0.7421944,  -1.2251937,   0.9038909 ],
        [ 0.7094993,  -0.67217445, -1.2523749,   1.2150501 ]]
    ])
    
    assert tf.is_tensor(encoderq_output), "Wrong type. Output must be a tensor"
    assert tuple(tf.shape(encoderq_output).numpy()) == (x.shape[0], x.shape[1], embedding_dim), \
           f"Wrong shape. We expected ({x.shape[0]}, {x.shape[1]}, {embedding_dim})"
    assert np.allclose(encoderq_output.numpy(), expected_output_1, atol=1e-1), "Wrong values case 1"
    
    # Case 2: training=True, with a mask
    mask = np.array([[[[1., 1., 1.]]], [[[1., 1., 0.]]]])
    encoderq_output = encoderq(x, training=True, mask=mask)
    expected_output_2 = np.array([
        [[ 0.17293282, -1.2640923,  -0.39698282,  1.4881423 ],
        [ 1.2104546,  -0.70949656, -1.2266788,   0.7257209 ],
        [ 0.85109115, -0.94624186, -1.0423193,   1.13747   ]],
        [[-1.1683232,  -0.8065115,   0.8559461,   1.1188885 ],
        [ 1.4513406,  -0.37653315, -1.3036361,   0.22882864],
        [ 1.036781,   -0.28892025, -1.5057343,   0.75787365]]
    ])
    assert np.allclose(encoderq_output.numpy(), expected_output_2, atol=1e-1), "Wrong values case 2"
    
    # Case 3, False, w a mask
    mask = np.array([[[[1., 1., 1.]]], [[[1., 1., 0.]]]])
    encoderq_output = encoderq(x, training=False, mask=mask)
    expected_output_3 = np.array([
        [[ 0.1758316,  -0.91559565, -0.825346,    1.56511   ],
        [ 0.9320458,   0.5653881,   0.17064969, -1.6680837 ],
        [ 1.2397449,  -0.8704194,  -1.0894277,   0.7201022 ]],
        [[ 1.617732,   -0.17120405, -0.33121377, -1.1153142 ],
        [ 1.1895713,  -0.83762133, -1.130274,    0.778324  ],
        [ 1.0139039,  -0.63975734, -1.3034768,   0.92933047]]
    ])
    assert np.allclose(encoderq_output.numpy(), expected_output_3, atol=1e-1), "Wrong values case 3"
    
    print("\033[92mAll tests passed")


    
def DecoderLayer_test(target, create_look_ahead_mask):
    
    num_heads=8
    tf.random.set_seed(10)
    
    decoderLayerq = target(
        embedding_dim=4, 
        num_heads=num_heads,
        fully_connected_dim=32, 
        dropout_rate=0.1, 
        layernorm_eps=1e-6)
    
    encoderq_output = tf.constant([[[-0.40172306,  0.11519244, -1.2322885,   1.5188192 ],
                                   [ 0.4017268,   0.33922842, -1.6836855,   0.9427304 ],
                                   [ 0.4685002,  -1.6252842,   0.09368491,  1.063099  ]]])
    
    q = np.array([[[1, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1]]]).astype(np.float32)
    
    look_ahead_mask = create_look_ahead_mask(q.shape[1])
    
    padding_mask = None
    out, attn_w_b1, attn_w_b2 = decoderLayerq(x=q, enc_output=encoderq_output, training=True, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
    
    assert tf.is_tensor(attn_w_b1), "Wrong type for attn_w_b1. Output must be a tensor"
    assert tf.is_tensor(attn_w_b2), "Wrong type for attn_w_b2. Output must be a tensor"
    assert tf.is_tensor(out), "Wrong type for out. Output must be a tensor"
    
    shape1 = (q.shape[0], num_heads, q.shape[1], q.shape[1])
    assert tuple(tf.shape(attn_w_b1).numpy()) == shape1, f"Wrong shape. We expected {shape1}"
    assert tuple(tf.shape(attn_w_b2).numpy()) == shape1, f"Wrong shape. We expected {shape1}"
    assert tuple(tf.shape(out).numpy()) == q.shape, f"Wrong shape. We expected {q.shape}"


    assert np.allclose(attn_w_b1[0, 0, 1], [0.4725226 ,  0.5274774 , 0.], atol=1e-2), "Wrong values in attn_w_b1. Check the call to self.mha1"
    assert np.allclose(attn_w_b2[0, 0, 1], [0.37197465 , 0.34991014 , 0.27811527], atol=1e-1),  "Wrong values in attn_w_b2. Check the call to self.mha2"
    
    print("\033[92mAll tests passed")
    
def Decoder_test(target, create_look_ahead_mask, create_padding_mask):
    tf.random.set_seed(10)
        
    num_layers=7
    embedding_dim=4 
    num_heads=3
    fully_connected_dim=8
    target_vocab_size=33
    maximum_position_encoding=6
    
    x_array = np.array([[3, 2, 1], [2, 1, 0]])

    
    encoderq_output = tf.constant([[[-0.40172306,  0.11519244, -1.2322885,   1.5188192 ],
                         [ 0.4017268,   0.33922842, -1.6836855,   0.9427304 ],
                         [ 0.4685002,  -1.6252842,   0.09368491,  1.063099  ]],
                        [[-0.3489219,   0.31335592, -1.3568854,   1.3924513 ],
                         [-0.08761203, -0.1680029,  -1.2742313,   1.5298463 ],
                         [ 0.2627198,  -1.6140151,   0.2212624 ,  1.130033  ]]])
    
    look_ahead_mask = create_look_ahead_mask(x_array.shape[1])
    
    decoderk = target(num_layers,
                    embedding_dim, 
                    num_heads, 
                    fully_connected_dim,
                    target_vocab_size,
                    maximum_position_encoding)
    x, attention_weights = decoderk(x=x_array, enc_output=encoderq_output, training=False, look_ahead_mask=look_ahead_mask, padding_mask=None)
    assert tf.is_tensor(x), "Wrong type for x. It must be a dict"
    assert np.allclose(tf.shape(x), tf.shape(encoderq_output)), f"Wrong shape. We expected { tf.shape(encoderq_output)}"
    
    keys = list(attention_weights.keys())
    assert type(attention_weights) == dict, "Wrong type for attention_weights[0]. Output must be a tensor"
    assert len(keys) == 2 * num_layers, f"Wrong length for attention weights. It must be 2 x num_layers = {2*num_layers}"
    assert tf.is_tensor(attention_weights[keys[0]]), f"Wrong type for attention_weights[{keys[0]}]. Output must be a tensor"
    shape1 = (x_array.shape[0], num_heads, x_array.shape[1], x_array.shape[1])
    assert tuple(tf.shape(attention_weights[keys[1]]).numpy()) == shape1, f"Wrong shape. We expected {shape1}" 
    
    print("\033[92mAll tests passed")
    
def Transformer_test(target, create_look_ahead_mask, create_padding_mask):
    
    tf.random.set_seed(10)

    num_layers = 6
    embedding_dim = 4
    num_heads = 4
    fully_connected_dim = 8
    input_vocab_size = 30
    target_vocab_size = 35
    max_positional_encoding_input = 5
    max_positional_encoding_target = 6

    trans = target(num_layers, 
                   embedding_dim, 
                   num_heads, 
                   fully_connected_dim, 
                   input_vocab_size, 
                   target_vocab_size, 
                   max_positional_encoding_input,
                   max_positional_encoding_target)

    # 0 is the padding value
    sentence_lang_a = np.array([[2, 1, 4, 3, 0]])
    sentence_lang_b = np.array([[3, 2, 1, 0, 0]])

    enc_padding_mask = create_padding_mask(sentence_lang_a)
    dec_padding_mask = create_padding_mask(sentence_lang_b)
    look_ahead_mask = create_look_ahead_mask(sentence_lang_a.shape[1])

    # Explicitly pass `training` as a keyword argument
    translation, weights = trans(
        input_sentence=sentence_lang_a,
        output_sentence=sentence_lang_b,
        training=True,  
        enc_padding_mask=enc_padding_mask,
        look_ahead_mask=look_ahead_mask,
        dec_padding_mask=dec_padding_mask
    )
    
    assert tf.is_tensor(translation), "Wrong type for translation. Output must be a tensor"
    shape1 = (sentence_lang_a.shape[0], max_positional_encoding_input, target_vocab_size)
    assert tuple(tf.shape(translation).numpy()) == shape1, f"Wrong shape. We expected {shape1}"
        
    
    keys = list(weights.keys())
    assert type(weights) == dict, "Wrong type for weights. It must be a dict"
    assert len(keys) == 2 * num_layers, f"Wrong length for attention weights. It must be 2 x num_layers = {2*num_layers}"
    assert tf.is_tensor(weights[keys[0]]), f"Wrong type for att_weights[{keys[0]}]. Output must be a tensor"

    shape1 = (sentence_lang_a.shape[0], num_heads, sentence_lang_a.shape[1], sentence_lang_a.shape[1])
    assert tuple(tf.shape(weights[keys[1]]).numpy()) == shape1, f"Wrong shape. We expected {shape1}" 

    # Explicitly pass `training` as a keyword argument
    translation, weights = trans(
        input_sentence=sentence_lang_a,  
        output_sentence=sentence_lang_b,
        training=False,  
        enc_padding_mask=enc_padding_mask,
        look_ahead_mask=look_ahead_mask,
        dec_padding_mask=dec_padding_mask
    )
    
    print(translation)
    
    print("\033[92mAll tests passed")
