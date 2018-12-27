# ERISK 2019: Early Detection of Signs of Self-harm

<h2> Useful links </h2>

eRisk CLEF: http://early.irlab.org/

Google docs link: https://drive.google.com/drive/u/0/folders/1OwOwZJjpG-574n4b5GAkOcaWN3izhAYa


<h2>Description</h2>

This is a new task in 2019. Essentially, it has the same format as T1 (but T2 has no training data!).

The challenge consists in performing a task on early risk detection of signs of self-harm. The challenge consists of sequentially processing pieces of evidence and detect early traces of self-harm as soon as possible. The task is mainly concerned about evaluating Text Mining solutions and, thus, it concentrates on texts written in Social Media. Texts should be processed in the order they were created. In this way, systems that effectively perform this task could be applied to sequentially monitor user interactions in blogs, social networks, or other types of online media.

The test collection for this task has the same format as the collection described in [Losada & Crestani 2016]. The source of data is also the same used for eRisk 2017 and 2018. It is a collection of writings (posts or comments) from a set of Social Media users. There are two categories of users, self-harm and non-self-harm, and, for each user, the collection contains a sequence of writings (in chronological order).

In 2019, we move from a chunk-based release of data (used in 2017 and 2018) to a item-by-item release of data. We set up a server that iteratively gives user writings to the participating teams. More information about the server is given here.

T2 has only a test stage (no training stage) and, therefore, we encourage participants to design their own unsupervised (e.g. search-based) strategies to detect possible cases of self-harm. The test stage will consist of a period of time where the participants have to connect to our server and iteratively get user writings and send responses.

Evaluation: The evaluation will take into account not only the correctness of the system's output (i.e. whether or not the user is depressed) but also the delay taken to emit its decision. To meet this aim, we will consider the ERDE metric proposed in [Losada & Crestani 2016] and other alternative evaluation measures.

The proceedings of the lab will be published in the online CEUR-WS Proceedings and on the conference website.

To have access to the collection all participants have to fill, sign and send a user agreement form (follow the instructions provided here). Once you have submitted the signed copyright form, you can proceed to register for the lab at CLEF 2019 Labs Registration site
