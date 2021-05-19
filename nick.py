
returnt = ''


def sav(nik):
    with open('nick.txt', 'a') as target:
        target.write(f'{nik}\n')



def writ(nik):
    with open('nick.txt', 'r') as target:
        for i in target:
            if nik in i:
                return returnt
        return 'Что-то новенькое!'
