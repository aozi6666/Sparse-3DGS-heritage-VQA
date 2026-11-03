# 成功与失败案例

看过真值后按规则及索引排序选出的定性示例，不代表随机样本，也不用于调参。仅整理字幕和采样位置，没有抽取或重新检查画面；不能仅凭这些记录断言因果机制。完整证据与单元哈希见 qualitative_cases.json。


## 226 · rescue_beyond_aligned · medium · Temporal Reasoning

视频：`IaTaaNinolU`；真值 D；无字幕 B → 原对齐 B → 连续 D。

Which of the following events happened after Polonius's death?

[" The norwegian army intended to invade a small area in polan.", " Hamlet's mother screamed out for help.", " Hamlet rebuked Ophelia's advances.", " Rosencrantz and Guildenstern escorted hamlet on a diplomatic mission."]

实际帧秒数：[0.0, 15.4154, 30.8308, 46.2462, 61.628233, 77.043633, 92.459033, 107.87443300000001, 123.289833, 138.705233, 154.087267, 169.502667, 184.918067, 200.33346699999998, 215.748867, 231.164267, 246.5463, 261.9617, 277.3771, 292.7925, 308.2079, 323.6233, 339.005333, 354.420733, 369.836133, 385.251533, 400.666933, 416.082333, 431.464367, 446.879767, 462.295167, 477.71056699999997]

原对齐字幕（以下至多1200字符，展示截取不影响模型输入）：

seeks to avenge his father's murder at
the hands of his uncle claudius
claudius has assumed the throne and
married his late brother's wife queen
whose character is he believes inferior
to that of his deceased father
when horatio tells hamlet of this
sighting
he's come from purgatory to demand
vengeance against his brother claudius
meanwhile claudius's chief advisor
polonius is concerned about his daughter
but he later becomes convinced that
ophelia's lack of affection is the
but when ophelia tries to flirt with
hamlet his mind is on his father's
friends rosencrantz and guildenstern to
discover the reason for hamlet's foul
that they ran into a troop of actors on
their way to the castle and invited them
if he seems rattled by the performance
it will prove his guilt
the country's
leadership claudius is reassured however
to do so
they need claudius's permission to march
play and even storms out
convincing hamlet of his guilt
hamlet nearly kills claudius but stopped
tries to convince his mother of
claudius's evil character
she screams out for help causing
polonius to scream as well
hamlet's friends rosencrantz and
guildenstern escort hamlet on a
instructing him to have hamlet killed
upo

连续字幕（展示至多1800字符）：

S5 [00:00:09.679 --> 00:00:23.519] about a danish prince named hamlet who seeks to avenge his father's murder at the hands of his uncle claudius the play is not only shakespeare's
S11 [00:00:25.439 --> 00:00:36.079] denmark has died and his brother claudius has assumed the throne and married his late brother's wife queen gertrude
S17 [00:00:38.879 --> 00:00:52.879] by his mother's marriage to claudius whose character is he believes inferior to that of his deceased father little does hamlet know the ghost of his
S24 [00:00:57.199 --> 00:01:06.240] night when horatio tells hamlet of this sighting hamlet decides to watch for the ghost
S30 [00:01:09.280 --> 00:01:22.479] addressing only hamlet the ghost says he's come from purgatory to demand vengeance against his brother claudius who murdered him by pouring poison into
S36 [00:01:25.439 --> 00:01:41.119] father's murder by killing claudius meanwhile claudius's chief advisor polonius is concerned about his daughter ophelia who loves hamlet but is confused
S42 [00:01:43.439 --> 00:01:54.039] stay away from hamlet but he later becomes convinced that ophelia's lack of affection is the reason for hamlet's melancholy
S48 [00:01:58.880 --> 00:02:10.959] claudius but when ophelia tries to flirt with hamlet his mind is on his father's murder and he rebukes her advances
S53 [00:02:10.959 --> 00:02:22.079] next claudius recruits hamlet's school friends rosencrantz and guildenstern to discover the reason for hamlet's foul mood
S60 [00:02:27.360 --> 00:02:39.360] rosencrantz and guildenstern tell hamlet that they ran into a troop of actors on their way to the castle and invited them along to perform for the court
S66 [00:02:41.440 --> 00:02:56.239] his father to see how claudius reacts if he seems rattled by the performance it will pro

模型原始输出：{"no_sub": "B", "aligned": "B", "continuous": "D"}


## 243 · rescue_beyond_aligned · medium · Information Synopsis

视频：`cMim0uU1yzA`；真值 B；无字幕 D → 原对齐 D → 连续 B。

Which summarizes the main focus of the video?

[" The effects of neurological disorders on brain function.", " What happens to the brain as we age.", " The effects of social and emotional education on brain development.", " Neurological development of the human brain from infancy to adulthood."]

实际帧秒数：[0.0, 16.96, 33.92, 50.88, 67.84, 84.8, 101.75999999999999, 118.72, 135.68, 152.64, 169.6, 186.56, 203.52, 220.48, 237.44, 254.4, 271.4, 288.36, 305.32, 322.28, 339.24, 356.2, 373.16, 390.12, 407.08, 424.04, 441.0, 457.96, 474.92, 491.88, 508.84, 525.8]

原对齐字幕（以下至多1200字符，展示截取不影响模型输入）：

radically inside your head your most
complex organ is constantly transforming
[Music]
plate around week four this folds over
onto itself forming a tube which
neurons at Birth your brain contains
adult but what you don't have is a lot
neurons some neurons become insulated by
a fatty substance called myelin these
number of copses per neuron is
2,500 but that's increases to
copses the speed at which they form
copses could be one reason children find
development from 3 until about 10 your
brain starts to remove connections it no
copses can be strengthened or weakened
depending on how they are
and want to read more of our coverage
why not take out a subscription to The
link by the time you become a teenager
your brain has stopped growing but it
parts of your brain develop at different
rates areas associated with reward like
rationality teens also experience major
changes in their lyic system the part of
difference in development between the
emotion-driven part of the brain and the
increased activity in The Venture medial
prefrontal cortex the part of the brain
friendships but it might also make them
more susceptible to social
prefrontal cortex increases connections
joining the emotional

连续字幕（展示至多1800字符）：

S6 [00:00:13.200 --> 00:00:24.599] changes radically inside your head your most complex organ is constantly transforming too beginning work before you're born
S13 [00:00:29.199 --> 00:00:38.879] brain [Music] your brain starts to form about 2 weeks
S19 [00:00:44.039 --> 00:00:56.960] called the neural plate around week four this folds over onto itself forming a tube which ultimately creates the nervous
S25 [00:00:59.920 --> 00:01:14.520] hundreds of nerve cells called neurons at Birth your brain contains about 100 billion neurons more than an
S31 [00:01:20.158 --> 00:01:32.720] you become an adult but what you don't have is a lot of shapses the point of contact where
S36 [00:01:35.079 --> 00:01:47.759] behavior of the neurons some neurons become insulated by a fatty substance called myelin these neurons known as white matter are able
S43 [00:01:52.560 --> 00:02:06.399] place when you're born the average number of copses per neuron is 2,500 but that's increases to approximately 15,000 per neuron over the
S48 [00:02:06.399 --> 00:02:19.760] neurons mature and create new copses the speed at which they form copses could be one reason children find it easier to learn things like languages
S55 [00:02:24.640 --> 00:02:40.760] lasting effects on their development from 3 until about 10 your brain starts to remove connections it no longer needs neuroscientists used to
S60 [00:02:40.760 --> 00:02:55.440] fixed level but now they understand that copses can be strengthened or weakened depending on how they are used this is known as
S68 [00:03:02.799 --> 00:03:12.599] information if you're enjoying this film and want to read more of our coverage why not take out a subscription to The Economist you'll receive daily and
S75 [00:03:17.959 --> 00:03:29.799] click the link by the time you

模型原始输出：{"no_sub": "D", "aligned": "D", "continuous": "B"}


## 257 · rescue_beyond_aligned · medium · Object Reasoning

视频：`e-P5IFTqB98`；真值 A；无字幕 D → 原对齐 D → 连续 A。

What indicates if we travel inside a black hole for quite a while?

[" The black hole has a super size massive.", " The black hole is rotating rapidly.", " The black hole is emitting visible light.", " The black hole has a charged event horizon."]

实际帧秒数：[0.0, 11.48, 22.96, 34.44, 45.92, 57.4, 68.88, 80.36, 91.8, 103.28, 114.75999999999999, 126.24, 137.72, 149.2, 160.68, 172.16, 183.64, 195.12, 206.6, 218.07999999999998, 229.56, 241.04, 252.52, 264.0, 275.44, 286.92, 298.4, 309.88, 321.36, 332.84000000000003, 344.32, 355.8]

原对齐字幕（以下至多1200字符，展示截取不影响模型输入）：

Black holes are one of the strangest things in existence.
maintaining a delicate balance between the two forces.
the heat and pressure at the core allow them to fuse heavier elements
Iron builds up at the center of the star
the star implodes.
as the star dies, in a super nova explosion.
If you looked at a black hole,
So we just see a black sphere
We're not sure what it is exactly.
nothing much would change for earth,
you seem to slow down as you approach the event horizon,
While from your perspective,
would means that gravity acts with millions of times more force on different parts of your body.
Two, you die a very quick death.
How soon you would die depends on the mass of the black hole.
As a rule of thumb,
and the diameter of an asteroid.
is S5 0014+81.
which is 47 times the distance from the sun to Pluto.
one of the virtual particles will be drawn into the black hole,
When it arrives at the mass of a large asteroid,
the black hole radiates away with the energy of billions of nuclear bombs in a huge explosion.
This is so long that when the last black hole radiates away,
there are loads more interesting ideas about black holes,

连续字幕（展示至多1800字符）：

S1 [00:00:00.000 --> 00:00:05.780] Black holes are one of the strangest things in existence. They don't seem to make any sense at all.
S10 [00:00:31.620 --> 00:00:38.680] pushes against gravity, maintaining a delicate balance between the two forces. As long as there is fusion in the core,
S14 [00:00:41.180 --> 00:00:49.800] But for stars with way more mass then our own sun the heat and pressure at the core allow them to fuse heavier elements until they reach iron.
S19 [00:00:54.580 --> 00:01:00.500] doesn't generate any energy. Iron builds up at the center of the star until it reaches a critical amount
S24 [00:01:06.380 --> 00:01:11.720] Within a fraction of a second, the star implodes. Moving at about the quarter of the speed of light,
S28 [00:01:14.380 --> 00:01:24.400] It's at this very moment that all the heavier elements in the universe are created, as the star dies, in a super nova explosion. This produces either a neutron star,
S32 [00:01:26.300 --> 00:01:34.800] the entire mass of the core collapses into a black hole. If you looked at a black hole, what you'd really be seeing is the event horizon.
S37 [00:01:40.440 --> 00:01:46.039] In other words, its impossible. So we just see a black sphere reflecting nothing.
S42 [00:01:52.300 --> 00:01:58.820] The singularity. We're not sure what it is exactly. A singularity may be indefinitely dense,
S51 [00:02:14.680 --> 00:02:21.440] If we were to swap the sun for an equally massive black hole, nothing much would change for earth, except that we would freeze to death, of course.
S56 [00:02:28.000 --> 00:02:33.900] from the outside, you seem to slow down as you approach the event horizon, so time passes slower for you.
S61 [00:02:37.900 --> 00:02:43.120] and disapear. While from your perspective, you can watch the rest of

模型原始输出：{"no_sub": "D", "aligned": "D", "continuous": "A"}


## 224 · subtitle_rescue · medium · Attribute Perception

视频：`4jmMWohs1XM`；真值 A；无字幕 B → 原对齐 A → 连续 A。

Which of the following methods of revealing the original color of the ancient art work is not mentioned in the video?

[" By imagining the color that fits in well with the art work.", " Ultraviolet light.", " Pigment analysis.", " Sampling some visible colors."]

实际帧秒数：[0.0, 11.052708, 22.063708, 33.116417, 44.127417, 55.180125, 66.191125, 77.243833, 88.296542, 99.307542, 110.36025000000001, 121.37125, 132.423958, 143.476667, 154.487667, 165.54037499999998, 176.551375, 187.604083, 198.615083, 209.667792, 220.72050000000002, 231.73149999999998, 242.784208, 253.795208, 264.847917, 275.900625, 286.911625, 297.964333, 308.975333, 320.028042, 331.039042, 342.09175]

原对齐字幕（以下至多1200字符，展示截取不影响模型输入）：

history you might be familiar with this
little sculpture is very famous original
lost a time before it was dug up again
in 1860s today it lives in the Vatican
supposed to look like this he's supposed
to look like this
Greece and Rome were really colorful
their buildings were full of intricate
of things like epic battles glimmering
gods and pretty flowers but today most
Hollywood's fault and our high school
textbooks fall but most of all it's this
who sculpted him and with many others
who made white marble statues during the
sculptures like this one
Lao kun and his sons they fell in love
white their paint had just faded after
being buried or left out in the open air
them they left their masterpieces bare -
and that style took over inspiring
historians reinforce this bias namely
this guy Johann Joaquin vinkle Minh he
ancient art in it he argued that statues
like this one the Apollo of Belvedere
is he went out of his way to ignore
obvious evidence of coloured marble and
near perfectly preserved frescoes
featured all sorts of colored statues
in Pompeii Benkelman claimed it was too
primitive to have been made by them but
remember Augustus when archeologists
rediscovered him in the 1860s

连续字幕（展示至多1800字符）：

S3 [00:00:05.879 --> 00:00:16.049] if you've ever had to study Roman history you might be familiar with this little sculpture is very famous original it's called the Augustus of primaporta
S8 [00:00:16.049 --> 00:00:36.909] during his reign as Emperor then it was lost a time before it was dug up again in 1860s today it lives in the Vatican Museums alongside a bunch of other famous sculptures but Augustus he's not supposed to look like this he's supposed to look like this [Music]
S16 [00:00:39.090 --> 00:00:59.170] let's get this out of the way ancient Greece and Rome were really colorful their buildings were full of intricate frescoes and elaborate mosaics and covered with violently painted statues of things like epic battles glimmering gods and pretty flowers but today most of us picture something more like this
S26 [00:00:59.880 --> 00:01:11.350] we're wrong but it's not our fault it's Hollywood's fault and our high school textbooks fall but most of all it's this guy's fault well not him he's just a
S31 [00:01:11.350 --> 00:01:21.670] the blame lies with Michelangelo the guy who sculpted him and with many others who made white marble statues during the Renaissance when European artists
S38 [00:01:25.180 --> 00:01:35.410] Michelangelo began studying Roman sculptures like this one Lao kun and his sons they fell in love with its lifelike figures dramatic scene
S43 [00:01:35.410 --> 00:02:06.220] sculptures like this weren't meant to be white their paint had just faded after being buried or left out in the open air for hundreds of years so when the Renaissance artist set out to imitate them they left their masterpieces bare - and that style took over inspiring generations of sculptors white marble became the norm along the way our historians reinforce this bias namely thi

模型原始输出：{"no_sub": "B", "aligned": "A", "continuous": "A"}


## 278 · subtitle_rescue · medium · OCR Problems

视频：`ce_uHxXsUWA`；真值 C；无字幕 A → 原对齐 C → 连续 C。

Which question is not included in the frequently asked questions displayed on the website?

[" How do I create a MyTrabelGov account?", " What does the \"Renew Passport\" button look like in my account?", " What is the processing time for renewals?", " Why don't I see an option to \"Renew Passport\" in my account?"]

实际帧秒数：[0.0, 18.351667, 36.669967, 55.021633, 73.3733, 91.72496699999999, 110.043267, 128.394933, 146.7466, 165.098267, 183.416567, 201.768233, 220.1199, 238.471567, 256.789867, 275.141533, 293.4932, 311.844867, 330.163167, 348.514833, 366.8665, 385.218167, 403.536467, 421.888133, 440.2398, 458.591467, 476.909767, 495.261433, 513.6131, 531.964767, 550.283067, 568.634733]

原对齐字幕（以下至多1200字符，展示截取不影响模型输入）：

renew their passports online in August and then again in October. I did not share information
it opens up for a limited amount of time every month. Once they get a certain number
At the time that this video was filmed, here's who qualifies to renew their passport online. If
in order to apply for your passport online. This is the homepage and it has changed a little bit
and you'll see that there are two options, whereas before there used to be one. I'm an adult renewing
Now, how do I know if I can renew online? It's laid out here pretty well, so I'm just going to
the age of 25 in order to be able to do this. Your recent passport was issued over 9 years,
birth. If you need to change any of these things, you'll have to apply through mail. You're not
a little quicker. Then again, you're applying for a regular passport, which is most of you.
you, it's not damaged and you've never reported it lost or stolen, and you do keep your recent
Then, very important, you need to be able to upload a digital passport photo,
and they will put it on a USB drive for you, which you can then upload to your application,
travel. These are the criteria as of the time of the filming of this video that you ne

连续字幕（展示至多1800字符）：

S2 [00:00:07.200 --> 00:00:18.840] August 2022, the State Department launched a pilot program that allowed a limited number of people to renew their passports online in August and then again in October. I did not share information
S7 [00:00:34.440 --> 00:00:43.740] it opens up for a limited amount of time every month. Once they get a certain number of applications, they close it out for a month and then open it again the following month. Once
S11 [00:00:53.280 --> 00:01:04.739] At the time that this video was filmed, here's who qualifies to renew their passport online. If this changes, I will publish an updated video, but if you don't see one, then this information still
S14 [00:01:11.220 --> 00:01:17.640] in order to apply for your passport online. This is the homepage and it has changed a little bit
S16 [00:01:22.020 --> 00:01:33.600] click on get a US passport, and now, since you're renewing you'll go to, I already have a passport, and you'll see that there are two options, whereas before there used to be one. I'm an adult renewing
S21 [00:01:48.600 --> 00:01:59.280] Now, how do I know if I can renew online? It's laid out here pretty well, so I'm just going to run through this, but if you want more time to read this, just head on over to the site like
S24 [00:02:05.100 --> 00:02:17.760] the age of 25 in order to be able to do this. Your recent passport was issued over 9 years, but less than 15 years from the date you plan to submit your application, and it's okay if your
S27 [00:02:23.520 --> 00:02:28.200] birth. If you need to change any of these things, you'll have to apply through mail. You're not
S30 [00:02:39.300 --> 00:02:50.160] then I would definitely apply via mail because there are a few different options to make it come a little quicker. Then again, you're

模型原始输出：{"no_sub": "A", "aligned": "C", "continuous": "C"}


## 272 · harm_vs_no_sub · medium · Object Reasoning

视频：`IGyx5UEwgtA`；真值 B；无字幕 B → 原对齐 C → 连续 C。

What is the example of avocados in the video meant to illustrate?

[" The court only decides cases with certainty.", " The courts are only deciding cases that are fully developed.", " The courts only rule on cases that are standing.", " The courts only rule on cases without controversy."]

实际帧秒数：[0.0, 13.5135, 27.027, 40.5405, 54.012292, 67.525792, 81.039292, 94.552792, 108.066292, 121.579792, 135.093292, 148.60679199999998, 162.078583, 175.592083, 189.105583, 202.619083, 216.132583, 229.646083, 243.159583, 256.673083, 270.144875, 283.658375, 297.171875, 310.685375, 324.198875, 337.712375, 351.225875, 364.739375, 378.211167, 391.724667, 405.238167, 418.751667]

原对齐字幕（以下至多1200字符，展示截取不影响模型输入）：

John Green, then Stan, then Brandon, then Zulaiha, then a bunch of independent contractors
Anyway, it's the strict adherence to hierarchy that makes CC HQ run like a well-oiled machine.
As you probably remember, because you're a smart rememberer of things, the American court
you realize there are 50 states, each with its own court system, it shouldn't be a surprise
On the federal side, the trial courts are the U.S. district courts. There are 94 of
Almost all criminal cases start in state courts. So if you don't like the result in a trial
most states have two levels of appellate courts, and the federal system has one. Appeals courts
regionally throughout the U.S.. You might guess if the region where your case is heard
court are before a single judge. There are four scenarios where the federal courts have
federal laws. These are pretty rare and rarely interesting.
in dispute are in different states, they might not even agree where to have the trial, and
court decisions. At least until the spring when the supreme court starts handing down
of appeals. If you lose there, you really really lost. When the court hears a case it's
are in fact lawyers. Although there's no constitutional requ

连续字幕（展示至多1800字符）：

S2 [00:00:07.450 --> 00:00:19.789] here at Crash Course we believe in hierarchy. First at the top you got me, then there's John Green, then Stan, then Brandon, then Zulaiha, then a bunch of independent contractors - wait I'm an independent contractor. But I'm still at the top! Anyway then the rest
S6 [00:00:24.350 --> 00:00:48.839] Anyway, it's the strict adherence to hierarchy that makes CC HQ run like a well-oiled machine. The same can be said for the U.S. court system. [Theme Music] As you probably remember, because you're a smart rememberer of things, the American court system is hierarchical which means any discussion of it cries out for a visual representation. Thought Bubble!
S12 [00:00:52.549 --> 00:00:59.799] you realize there are 50 states, each with its own court system, it shouldn't be a surprise that the vast majority of cases start out in state courts. Of course most cases never
S15 [00:01:03.309 --> 00:01:08.900] outside of court. How's that for efficiency. On the federal side, the trial courts are the U.S. district courts. There are 94 of
S19 [00:01:17.230 --> 00:01:27.160] But you can only start in the district courts if your case meets certain important criteria. Almost all criminal cases start in state courts. So if you don't like the result in a trial court, and you have a reasonable claim that there was something procedurally wrong with
S23 [00:01:29.940 --> 00:01:36.850] Notice that if you start in state courts, you usually have more chances for appeal because most states have two levels of appellate courts, and the federal system has one. Appeals courts
S27 [00:01:44.520 --> 00:01:54.340] Federal appeals courts are called circuit courts, and there are 12 of them distributed regionally throughout the U.S.. You might guess if the region where your c

模型原始输出：{"no_sub": "B", "aligned": "C", "continuous": "C"}


## 358 · harm_vs_no_sub · medium · Object Recognition

视频：`2Gg4OQo7-zA`；真值 A；无字幕 A → 原对齐 D → 连续 D。

In Germany vs Argentina, Which player scores the first goal?

[" Germany's player number 19.", " Germany's player number 9.", " Argentina's player number 10.", " Argentina's player number 19."]

实际帧秒数：[0.0, 27.0, 54.0, 81.03999999999999, 108.03999999999999, 135.04, 162.04, 189.04, 216.07999999999998, 243.08, 270.08, 297.08, 324.08, 351.12, 378.12, 405.12, 432.12, 459.12, 486.12, 513.16, 540.16, 567.16, 594.16, 621.16, 648.2, 675.2, 702.2, 729.2, 756.2, 783.24, 810.24, 837.24]

原对齐字幕（以下至多1200字符，展示截取不影响模型输入）：

foreign
[Applause]
and whilst it was a good save by taparel
[Applause]
[Applause]
hesitation from taparel
after years of waiting and wondering
it's why
[Music]
fine save from Marcos
this or that now Rivaldo oh and can't
for the second time in the space of a
foreign
[Music]
for matarazzi
the moment of near instant redention
so he was about to provide his career
[Music]
it's career but never like this he's
scores
Italy win
who should have opened the story
[Music]
[Applause]
the French are screaming at ball
play with the world at his feet
unless he turns it out to Alvarez who
gets it through
rounds down and that's got to be a
penalty
in a Flash in a heartbeat
hey
is the man
saved by Martinez
[Music]
Kings of football for the third time
and the tears will fall
subscribe now
[Music]

连续字幕（展示至多1800字符）：

S1 [00:00:00.000 --> 00:00:32.700] foreign [Music] [Applause] to the T
S9 [00:00:37.340 --> 00:00:59.719] now give us here's an opportunity and whilst it was a good save by taparel [Applause] Ronaldo
S18 [00:01:10.680 --> 00:01:25.619] size of red people hesitated [Applause] hesitation from taparel so there was
S27 [00:01:42.079 --> 00:01:56.180] exit three and secures the World Cup after years of waiting and wondering now tears of pure joy
S33 [00:02:08.419 --> 00:02:20.520] slipping it through Ronaldo's in and it's why [Music] we've got our video beautiful form over
S43 [00:02:35.400 --> 00:02:49.220] stunning swerving strike and an equally fine save from Marcos this or that now Rivaldo oh and can't spilled it Ronaldo to make the
S52 [00:03:06.260 --> 00:03:15.180] for the second time in the space of a dozen second half Oliver beerhoff Spot
S61 [00:03:31.260 --> 00:03:45.300] feed in the final for the fourth time foreign [Music] ER who takes it on and goes down penalty
S69 [00:03:54.000 --> 00:04:05.000] from the right side of the line for matarazzi the moment of near instant redention
S79 [00:04:24.840 --> 00:04:33.680] what to say so he was about to provide his career with the perfect endings
S88 [00:04:54.300 --> 00:05:07.580] the attorney [Music] it's career but never like this he's done it now emphatically one one
S97 [00:05:20.580 --> 00:05:28.460] out scores no Keepers keeping that one out
S105 [00:05:45.180 --> 00:05:57.290] so so close now Italy win your scores he went for the blast
S121 [00:07:38.539 --> 00:07:47.380] who should have opened the story it up close with the take against the
S128 [00:07:56.880 --> 00:08:06.480] but their water Target area [Music]
S135 [00:08:30.760 --> 00:08:39.380] [Applause] in the World Cup final
S141 [00:08:57.980 --> 00:09:

模型原始输出：{"no_sub": "A", "aligned": "D", "continuous": "D"}


## 404 · harm_vs_no_sub · medium · Counting Problem

视频：`hYTYb5sGznw`；真值 C；无字幕 C → 原对齐 B → 连续 A。

How many yeses do the performers get?

[" 3.", " 2.", " 4.", " 1."]

实际帧秒数：[0.0, 13.947267, 27.861167, 41.808433, 55.722333, 69.6696, 83.616867, 97.530767, 111.47803300000001, 125.4253, 139.3392, 153.28646700000002, 167.200367, 181.147633, 195.0949, 209.0088, 222.956067, 236.869967, 250.817233, 264.7645, 278.6784, 292.625667, 306.539567, 320.486833, 334.4341, 348.348, 362.295267, 376.242533, 390.156433, 404.1037, 418.0176, 431.964867]

原对齐字幕（以下至多1200字符，展示截取不影响模型输入）：

[Applause]
couple
[Applause]
arm day
is the little girl over there is
[Applause]
yes I'm sorry she gave me a beautiful
come on down come on down
how long have you guys been performing
looks like you're gonna do that
ventriloquist act that you did
[Applause]
[Music]
oh
my God
[Applause]
[Applause]
[Applause]
[Music]
[Applause]
[Applause]
[Music]
foreign
[Applause]
the fact that you're able to hang
yourself outside sideways with your wife
presenting your judges with the
firstborn to win the contest
Heidi I mean I see muscles but most
importantly I see so much grace it was
before truly unique very very special
moment thank you Simon I don't love
least four years to Perfection it and to
make it work
Heidi a really really really big scene
yeah
give the baby back
[Music]
[Applause]

连续字幕（展示至多1800字符）：

S3 [00:00:08.220 --> 00:00:21.859] name is Colombia [Applause] Colombia
S8 [00:00:25.380 --> 00:01:00.300] you guys I'm guessing you guys are a couple [Applause] they've been together for nine years wow I was gonna say they met at the gym on arm day is the little girl over there is [Applause] oh sorry this morning [Applause] yes I'm sorry she gave me a beautiful painting outside of the theater this
S22 [00:01:06.180 --> 00:01:13.280] come on down come on down how long have you guys been performing together
S27 [00:01:25.140 --> 00:01:39.380] [Applause] looks like you're gonna do that ventriloquist act that you did
S31 [00:01:45.360 --> 00:02:01.159] [Music] [Applause] [Music]
S35 [00:02:18.300 --> 00:02:20.599] oh
S37 [00:02:28.060 --> 00:02:38.379] [Applause] my God [Applause] [Music]
S44 [00:02:57.210 --> 00:03:06.399] [Applause]
S45 [00:03:16.820 --> 00:03:47.380] [Applause] [Music] [Applause] [Music]
S50 [00:03:47.390 --> 00:04:38.040] [Music] [Applause] [Music] foreign [Applause] [Music]
S59 [00:04:49.800 --> 00:04:58.860] the fact that you're able to hang yourself outside sideways with your wife standing on you I've never seen that and
S64 [00:05:01.320 --> 00:05:13.740] surprising me and just the move of presenting your judges with the firstborn to win the contest is a move that has never been done
S70 [00:05:14.840 --> 00:05:42.120] your gift Heidi I mean I see muscles but most importantly I see so much grace it was So Graceful how you it was a beautiful day and sideways we have never seen this before truly unique very very special moment thank you Simon I don't love these kind of Acts having said that that
S81 [00:05:54.360 --> 00:06:35.880] she stands on top of him it took them at least four years to Perfection it and to make it work amazing Howie your boat se

模型原始输出：{"no_sub": "C.", "aligned": "B.", "continuous": "A."}


## 462 · harm_vs_aligned · medium · Counting Problem

视频：`FMdEkj5HEqg`；真值 B；无字幕 A → 原对齐 B → 连续 C。

How many times is the little boy on the right side of the planet when he visited the planet in his rocket?

[" 3.", " 4.", " 5.", " 2."]

实际帧秒数：[0.0, 22.133333, 44.3, 66.433333, 88.6, 110.733333, 132.9, 155.033333, 177.2, 199.333333, 221.5, 243.633333, 265.766667, 287.933333, 310.066667, 332.233333, 354.366667, 376.53333299999997, 398.666667, 420.833333, 442.96666700000003, 465.1, 487.266667, 509.4, 531.566667, 553.7, 575.866667, 598.0, 620.166667, 642.3, 664.466667, 686.6]

原对齐字幕（以下至多1200字符，展示截取不影响模型输入）：

что ты делаешь с этим биноклем
[Музыка]
хорошо, я вижу
Хорошо
и сколько планет в
отличный
Вы знаете
да, конечно, меня это очень интересует
это очень нестабильно
Я не знал этого и почему
на самом деле ученые так думают
на его поверхности была жидкая вода
и какова его температура
да, мы идем туда
Земле необходимо 365 дней
это интересно
это четвертая планета нашего
Солнце
[Музыка]
да, это слишком долго
теперь пойдем на Сатурн
солнечный
она известна своим великолепным
обойти солнце
она колеблется от - 226 до - 197 градусов
Цельсия
это также рассматривается как
ледяной гигант
большое спасибо за все это
если вам понравилось это видео нажмите
на кнопке лайк

连续字幕（展示至多1800字符）：

S6 [00:00:21.439 --> 00:00:26.720] что ты делаешь с этим биноклем в середине ночи
S13 [00:00:39.120 --> 00:00:49.990] Солнечная система [Музыка] хорошо, я вижу скажи мне сначала, что это такое
S21 [00:00:58.559 --> 00:01:09.799] наша планета Земля Хорошо и сколько планет в Солнечная система
S27 [00:01:21.060 --> 00:01:33.439] Уран и Нептун отличный Вы знаете характеристики каждого из этих
S35 [00:01:43.079 --> 00:01:55.880] и посетите каждый из них да, конечно, меня это очень интересует Идеальный
S47 [00:02:53.099 --> 00:03:04.819] градусов Цельсия это очень нестабильно да и там год длится 88 дней
S58 [00:03:37.019 --> 00:03:48.900] Это правда Я не знал этого и почему просто потому, что они
S64 [00:03:56.220 --> 00:04:09.200] когда-то, возможно, она была на самом деле ученые так думают на его поверхности была жидкая вода но сейчас нет
S71 [00:04:17.399 --> 00:04:30.780] это занимает около 225 земных дней и какова его температура это самая жаркая планета в мире
S79 [00:04:44.580 --> 00:04:50.780] Земля да, мы идем туда
S81 [00:05:04.440 --> 00:05:15.259] наша земля находится на третьей позиции Земле необходимо 365 дней вращаться вокруг Солнца
S88 [00:05:27.419 --> 00:05:39.800] и наоборот это интересно расскажи мне о марте
S92 [00:05:52.979 --> 00:05:57.800] это четвертая планета нашего Солнечная система
S109 [00:06:56.400 --> 00:07:07.560] а как насчет его революции вокруг Солнце [Музыка] хотя эти дни самые
S116 [00:07:15.259 --> 00:07:24.620] ох, это слишком долго да, это слишком долго теперь пойдем на Сатурн
S122 [00:07:39.960 --> 00:07:48.560] это самая большая планета в системе солнечный она известна своим великолепным наши мечи
S141 [00:08:46.440 --> 00:08:57.140] это время вращения обойти солнце Урану нужен период в 84 года
S146 [00:09:04.380 --> 00:09:20.700] и к

模型原始输出：{"no_sub": "A.", "aligned": "B", "continuous": "C."}


## 470 · harm_vs_aligned · medium · Counting Problem

视频：`cVIfe0Gxa64`；真值 B；无字幕 B → 原对齐 B → 连续 D。

How many cars finished the race in the video?

[" 23.", " 21.", " 25.", " 29."]

实际帧秒数：[0.0, 9.24, 18.48, 27.68, 36.92, 46.16, 55.4, 64.6, 73.84, 83.08, 92.32, 101.56, 110.75999999999999, 120.0, 129.24, 138.48, 147.68, 156.92000000000002, 166.16, 175.4, 184.6, 193.84, 203.07999999999998, 212.32, 221.56, 230.76, 240.0, 249.24, 258.48, 267.68, 276.92, 286.15999999999997]

原对齐字幕（以下至多1200字符，展示截取不影响模型输入）：

des Jahres gemeldet in der ADAC GT4
Germany ein starker Regenschauer kurz
dem Safety Guard nach zwei
Einführungsrunden die Freigabe Hugo
weiteren Aston Martin dahinter generell
drei Aston Martin an der Spitze
der Halbzeit führende
möchte er gerne sein
der Nummer 4 von bcmc Motorsport powert
bei Eastside Motor aber das Qualifying
Plätze nach hinten wurden sie verschoben
wegen eines Vergehens im ersten freien
hinten im gesamten Feld erneut einsetzen
der Regen nachdem viele beim flickboxen
Bedingungen und wir sehen den ersten
Abflug nach gut 11 Minuten eine
Mücke Motorsport Mercedes und mit der
Nummer 30 Max Kronberg im Avia B&S
werden deshalb zunächst einmal der
Rennen Abbruch erneuter Start hinter dem
die Strecke fängt an so langsam ab zu
trocknen das sieht man an dem heller
gefaltet um jede einzelne Position zu
sehen hier aus dem Blickwinkel von
in einem drehre hinein gejagt ist
resixty direkt den Fahrtrichtung mit Gas
zwischen David Jan und
bulatov um die vierte Position hinter
Rennen Geschehen David Jan mit einem
Reifen Schaden hinten rechts fällt erst
Boxenstopp Fenster geöffnet
in die Boxengasse um das Auto an seinen
das ist
ein Meter aus Mosambik mit einem materia
liegende bul

连续字幕（展示至多1800字符）：

S1 [00:00:00.860 --> 00:01:27.299] 29 Fahrzeuge beim fünften Wertungslauf des Jahres gemeldet in der ADAC GT4 Germany ein starker Regenschauer kurz vor Rennbeginn nötig die Teams dazu auf Regenreifen zu wechseln der Start hinter dem Safety Guard nach zwei Einführungsrunden die Freigabe Hugo Sachse im Aston Martin der Pohlmann bestimmt hier das Tempo bandeur in einem weiteren Aston Martin dahinter generell drei Aston Martin an der Spitze der Halbzeit führende möchte er gerne sein Dennis bonatov zusammen mit seinen Partner Marcel Lennartz im Fahrzeug mit der Nummer 4 von bcmc Motorsport powert bei Eastside Motor aber das Qualifying als schnellste eine grippstrafe drei Plätze nach hinten wurden sie verschoben wegen eines Vergehens im ersten freien Training und dann heißt es harte Positionskämpfe bestreiten von vorne bis hinten im gesamten Feld erneut einsetzen der Regen nachdem viele beim flickboxen Stopp mal kurz aufs links gewechselt haben es sind ganz schwierige Bedingungen und wir sehen den ersten Abflug nach gut 11 Minuten eine Kollision zwischen Alexander Komma in Mücke Motorsport Mercedes und mit der Nummer 30 Max Kronberg im Avia B&S Motorsport Porsche Cayman die
S35 [00:01:28.500 --> 00:02:13.680] Streckenbegrenzung muss wieder aufgebaut werden deshalb zunächst einmal der Rennen Abbruch erneuter Start hinter dem Safety kam Zeit läuft weiter wir sind noch vor dem unintelligible Boxenstop die Strecke fängt an so langsam ab zu trocknen das sieht man an dem heller werdenden Asphalt auf der am meisten befahrenen Linie und es wird herzhaft gefaltet um jede einzelne Position zu sehen hier aus dem Blickwinkel von Dennis bulatoff Müller Matzen wird dann in einem drehre hinein gejagt ist resixty direkt den Fahrtrichtung mit Gas die Fahrt wieder aufnehmen verliert er ein paa

模型原始输出：{"no_sub": "B", "aligned": "B", "continuous": "D"}
