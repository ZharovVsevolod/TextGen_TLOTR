import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import itertools

from sklearn.model_selection import train_test_split

import torch
from torch import nn

import youtokentome as yttm

from utils.poetry import *
from utils.base import *
from utils.transformer_tools import *

init_random_seed(1702)

#---------------------------------

# Предобработка изначального файла, кодировка, уничтожение пустых строк
with open("datasets/text.txt", encoding="utf-8") as input_file:
    data = input_file.read().split('\n')
    data = [value for value in data if value != ""]
# И сохранение получившегося текста в отдельный файл
with open("datasets/text_red.txt", "w", encoding="utf-8") as f:
    for line in data:
        f.write(f"{line}\n")

#---------------------------------

# Загрузка подготовленного текста, деление его на определённые длины предложения
data = load_chunks("./datasets/text_red.txt", chunk_size=300)
# Деление исходной выборки на два сета данных для предварительного обучения 
# нейросети-трансформера и генеративно-состязательной сети в соотношении 2:1
ALL_LEN = len(data)
print(ALL_LEN)
len_for_steps = int(ALL_LEN / 3)
data_for_GAN = data[len_for_steps * 2:]
data = data[:len_for_steps * 2]

#---------------------------------

# Деление выборки для предварительного обучения на обучающую и тестовую в соотношении 7:3
data_train, data_test = train_test_split(data, test_size=0.3)
print('Размер обучающей выборки', len(data_train))
print('Размер тестовой выборки', len(data_test))

#---------------------------------

# Токенизация с помощью библиотеки youtokentome
BPE_MODEL_FILENAME = './models/My_models/text_bpe_1000.yttm'

# Библиотека работает только с отдельно записанным txt файлом, поэтому текст, на основе 
# которого будет производиться токенизация, отдельно записывается в файл
TRAIN_TEXTS_FILENAME = './datasets/text_train.txt'
save_texts_to_file(data_train, TRAIN_TEXTS_FILENAME)
yttm.BPE.train(data=TRAIN_TEXTS_FILENAME, vocab_size=1000, model=BPE_MODEL_FILENAME)

# Загрузка словаря токенов и его просмотр
tokenizer = yttm.BPE(BPE_MODEL_FILENAME)
print(' '.join(tokenizer.vocab()))

#---------------------------------

# Токенизация текста для предварительного обучения нейросети-трансформера
train_token_ids = tokenizer.encode(data_train, bos=True, eos=True)
test_token_ids = tokenizer.encode(data_test, bos=True, eos=True)
# Токенизация выборки для генеративно-состязательной сети
token_GAN = tokenizer.encode(data_for_GAN, bos=True, eos=True)

# Проверка, что в другом тексте незнакомых токенов не появилось
unknown_subwords_in_test = sum(1 for text in test_token_ids for token_id in text if token_id == 1)
print('Количество случаев с неизвестными n-граммами символов в валидационной выборке', unknown_subwords_in_test)
# И для выборки для генеративно-состязательной сети
unknown_subwords_in_test = sum(1 for text in token_GAN for token_id in text if token_id == 1)
print('Количество случаев с неизвестными n-граммами символов', unknown_subwords_in_test)

#---------------------------------

# Построение гистограммы распределения длины предложений после токенизации для того,
# чтобы определить оптимальный размер предложения уже после токенизации
plt.hist([len(sent) for sent in train_token_ids], bins=30)
plt.title('Распределение длин фрагментов в токенах')
plt.yscale('log')

#---------------------------------

# Загрузка текста с помощью модулей Dataset
CHUNK_LENGTH = 100
train_dataset = LanguageModelDataset(train_token_ids, chunk_length=CHUNK_LENGTH)
test_dataset = LanguageModelDataset(test_token_ids, chunk_length=CHUNK_LENGTH)

# Установка "начала" и "конца" предложения на основе ранее определённого CHUNK_LENGTH
start_length = 20
end_length = 50
GAN_dataset = Dataset_for_GAN(token_GAN, start_len=start_length, end_len=end_length)

#---------------------------------

# Создание модели для генерации текста с необходимыми параметрами
model = Language_Model(
    vocab_size = tokenizer.vocab_size(),
    embedding_size = 256,
    backbone = Transformer_Encoder(
        nn.TransformerEncoderLayer(
            d_model = 256,
            nhead = 16,
            dim_feedforward = 512,
            dropout = 0.3
        ),
        num_layers=5
    ),
    emb_dropout = 0.2
)
print('Количество параметров', get_params_number(model))

#---------------------------------

# Обучение модели
loss_history_all, best_model, optimizer = train_loop(
    model,
    train_dataset,
    test_dataset,
    lm_cross_entropy,
    # save_path="./models/My_models/War/war_2/",
    need_to_save=False,
    lr=1e-4,
    epoch_n=2000,
    batch_size=128,
    device='cuda',
    early_stopping_patience=50,
    # early_optimizer_SD=optimizer_SD,
    max_batches_per_epoch_train=1000,
    max_batches_per_epoch_val=1000,
    lr_scheduler_ctor=lr_scheduler,
    need_to_gen=True,
    tokenizer=tokenizer,
    pharse="For the king and "
)

#---------------------------------

# Блок сохранений
# Сохранение модели и оптимизатора
torch.save(best_model.state_dict(), "./models/Time/2/Don.pth")
torch.save(optimizer.state_dict(), './models/Time/2/Don_opt.pth')

# Сохранение истории значений loss-функции
loss_history_all = [str(line) for line in loss_history_all]
save_texts_to_file(loss_history_all, "./models/Time/2/Don_1.txt")

#---------------------------------

# Загрузка модели и токенизатора
optimizer_SD = torch.load('./models/Time/2/Don_opt.pth')
model.load_state_dict(torch.load('./models/Time/2/Don.pth'))

#---------------------------------

# Создание класса-генератора предложений с помощью лучевого поиска на основе языковой модели
beam_generator = BeamGenerator(model, tokenizer)

# Генерация предложений
answers = []
for i in range(5):
    beam_gen_variants = beam_generator('Сегодня Солнце', beamsize=5, return_hypotheses_n=1)
    for score, pred_txt in beam_gen_variants:
        answers.append(pred_txt)

for ans in answers:
    print(ans)

#---------------------------------

# Теперь генеративно-состязательная сеть

#---------------------------------

# Создание дискриминатора
D_model = LSTM_Discriminator(100, 256, tokenizer.vocab_size())
print('Количество параметров дискриминатора', get_params_number(D_model))

# Определение генератора
G_model = model
optimizer_SD_G = optimizer_SD

#---------------------------------

# Обучение с помощью генеративно-состязательной сети
batch_size = 64

model_D, model_G, optimizer_D, optimizer_G, loss_history_D, loss_history_G = train_loop_GAN(
    D_model,
    G_model,
    GAN_dataset,
    lm_b_cross_entropy,
    lr_D=1e-2,
    lr_G=1e-4,
    epoch_n=1000,
    batch_size=batch_size,
    device='cuda',
    # early_optimizer_SD_D=optimizer_SD_D,
    early_optimizer_SD_G=optimizer_SD_G,
    tokenizer=tokenizer,
    need_to_gen=True,
    end_chunk=end_length,
    pharse="For the king and "
)

#---------------------------------

# Сохранение всех моделей и их оптимизаторов
torch.save(model_D.state_dict(), './models/GAN/3/model_D.pth')
torch.save(model_G.state_dict(), './models/GAN/3/model_G.pth')
torch.save(optimizer_D.state_dict(), './models/GAN/3/optimizer_D.pth')
torch.save(optimizer_G.state_dict(), './models/GAN/3/optimizer_G.pth')

# Сохранение истории значений loss-функций
with open("./models/GAN/3/loss_D.txt", 'w') as outf:
    for num in loss_history_D:
        outf.write(str(num))
        outf.write("\n")

with open("./models/GAN/3/loss_G.txt", 'w') as outf:
    for num in loss_history_G:
        outf.write(str(num))
        outf.write("\n")

#---------------------------------

# Создание класса-генератора предложений с помощью лучевого поиска на основе языковой модели
beam_generator = BeamGenerator(model_G, tokenizer)

def gen_some_sent(gen_model, phrase, N, beam_size, need_reweight=False, temperature=0, alpha=0):
    """
    Функция для генерации нескольких различных продолжений фразы 
    с возможностью настройки параметров температуры и сглаживания меток
    """
    answers = []
    for i in range(N):
        beam_gen_variants = gen_model(
            phrase, beamsize=beam_size, return_hypotheses_n=1, 
            need_reweight=need_reweight, temperature=temperature, alpha=alpha
        )
        for score, pred_txt in beam_gen_variants:
            answers.append(pred_txt)
    return answers

def test_gen(gen_model, text, temperatures, alphas):
    """
    Фукнция для перебора комбинаций температуры и сглаживания для генерации продолжения фразы
    """
    all_test = itertools.product(temperatures, alphas)
    for t, a in all_test:
        print(f"Температура = {t}, альфа = {a}")
        answers = gen_some_sent(gen_model, text, 5, 10, True, t, a)
        for ans in answers:
            print(ans)
        print("-----")

#---------------------------------

temps = [0, 0.2, 0.5, 0.8]
alps = [0, 0.1, 0.3, 0.5]

test_gen(beam_generator, "Today we are", temps, alps)