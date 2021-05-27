
returnt = ''


def save(nik):
    """
    nik - NickName пользователя
    Записывает NickName в файл.
    """
    with open('nick.txt', 'a') as target:
        target.write(f'{nik}\n')



def read(nik):
    """
    nik - NickName пользователя.
    Читает файл с NickName пользователей.
    return - если NickName найден возвращает пустую строку.
    """
    with open('nick.txt', 'r') as target:
        for i in target:
            if nik in i:
                return returnt
        return 'Что-то новенькое!'
