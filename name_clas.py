
g = {0: """\t\t\tФизическая линия сердца - Человек открытый, не стесняющийся выражать даже самые потаенные чувства. По натуре он человек
теплый, любящий и заботливый. Когда с ним случаются неприятности, он не позволяет себе впадать в отчаяние и быстро
возвращается к нормальной жизни.\n""",
1: """\t\t\tДуховная линия сердца - Духовная линия сердца указывает на то, что человек не способен легко и непринужденно
обсуждать свои личные чувства и переживания.\n""",
2: """\t\t\tЛиния сердца заканчивающаяся между пальцами юпитера и сатурна - Человек, на руке которого линия сердца
заканчивается между указательным и средним пальцами, реалистично относится к своей личной жизни, не предъявляя
завышенных требований и не страдая от несбывшихся ожиданий.\n""",
3: """\t\t\tРаздвоенная линия сердца - Такие линии встречаются на руках людей, чья душевная организация очень сложна и
 многогранна. Эти люди способны видеть и понимать проблемы комплексно.\n""",
4: """\t\t\tЦепочка на линии сердца - Цепочки на линии сердца символизируют эмоциональное напряжение — частые душевные
взлеты и падения. Кресты и разрывы на линии указывают на эмоциональную потерю — окончание любовных взаимоотношений,
 возможно, в результате смерти партнера. Островок указывает на период депрессии и подавленного состояния.\n""",
5: """\t\t\tДвойная линия сердца - Если на руке человека присутствует двойная линия сердца, это говорит о том, что его
избранник будет исключительно заботливым и преданным.\n""",
6: """\t\t\tворческая линия ума - Человек старается подходить творчески ко всему что он делает, максимально используя
воображение.\n""",
7: """\t\t\tПрактичная линия ума - Человек применяет практичный и простой подход ко всему, чем занимается. Не склонен полагаться
 на слепую веру, предпочитая держать ситуацию под своим личным контролем. Ему нравится во всем докапываться до сути
 и самостоятельно принимать решения.\n""",
8: """\t\t\tПисательская резвилка линии ума - eе присутствие на ладони говорит о том, что человек обладает ярким, живым воображением
и часто рождает оригинальные идеи, которые впоследствии может реализовать на практике.\n""",
9: """\t\t\tЛиния ума, отражающая большие материальные запроы - В отдельных случаях такая линия ума может являться своего
рода благоприятным знаком, но, как правило, она указывает на человека, одержимого идеей обогащения до такой степени,
 что он не в состоянии остановиться.\n""",
10: """\t\t\tЛиния ума, удаленная от линии жизни - Чем дальше от линии жизни начинается линия ума, тем независимее и
свободолюбивее будет человек.\n""",
11: """\t\t\tЛиния ума, исходящая из линии жизни - Если в начале линия ума соприкасается с линией жизни,
 это указывает на осторожного, предусмотрительного человека, который сначала думает, а только потом делает.\n""",
12: """\t\t\tОчень длинная линия ума - Длинная линия ума указывает на разносторонне развитого человека, отличающегося
 обилием интересов и быстротой мышления .\n""",
13: """\t\t\tКороткая линия ума - Люди с короткой линией ума прямолинейны и практичны в помыслах и поступках.
 Они стараются как можно быстрее приступить к работе и выполнить ее, не теряя понапрасну времени.\n""",
14: """\t\t\tЛиния жизни обнимающая большойо палец - Такому человеку следует уделять больше внимания состоянию своего тела,
заниматься физическими упражнениями, и он станет более выносливым и энергичным.\n""",
15: """\t\t\tЛиния жизни полностью пересекающая ладонь - Этот человек очень вынослив, жизнелюбив и энергичен. Если он занимается делом,
 которое доставляет ему удовольствие, то не чувствует усталости и не испытывает потребности в отдыхе.\n""",
16: """\t\t\tЛиния сестра Эта линия всегда является исключительно благоприятным знаком. Если она расположена ближе к концу
 линии жизни, это говорит о том, что человек будет вести активный образ жизни даже в глубокой старости.\n"""}






def boxes(lis):
    f = [g[i] for i in lis]
    return '\n'.join(f)

#print(boxes([2, 3, 4]))