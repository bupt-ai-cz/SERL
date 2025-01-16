import os

def split(root="./",domain="",shot=5):
    labeled_target_images=open(os.path.join(root,"labeled_target_images_"+domain+"_3.txt"),'r').readlines()

    unlabeled_target_images=open(os.path.join(root,"unlabeled_target_images_"+domain+"_3.txt"),'r').readlines()

    validation_target_images=open(os.path.join(root,"validation_target_images_"+domain+"_3.txt"),'r').readlines()

    new_labeled=open(os.path.join(root,"labeled_target_images_"+domain+"_"+str(shot)+".txt"),'w')
    new_labeled.writelines(labeled_target_images)
    cnt_labeled=len(labeled_target_images)

    new_unlabeled=open(os.path.join(root,"unlabeled_target_images_"+domain+"_"+str(shot)+".txt"),'w')
    cnt_unlabeled=0
    class2images={}

    for line in unlabeled_target_images:
        [image,label]=line.split(' ')

        if line not in validation_target_images:
            if label not in class2images:
                class2images[label]=1
                new_labeled.writelines([line])
                cnt_labeled+=1
            elif class2images[label]<shot-3 :
                class2images[label]=class2images[label]+1
                new_labeled.writelines([line])
                cnt_labeled+=1
            else:
                new_unlabeled.writelines([line])
                cnt_unlabeled+=1
        else:
            new_unlabeled.writelines([line])
            cnt_unlabeled += 1

    print("len labeled {}, unlabeled {} val{} newlabeled {} ,new_unlabeled{}".format(len(labeled_target_images),
            len(unlabeled_target_images),len(validation_target_images),cnt_labeled,cnt_unlabeled))

for shot in [5,10]:
    split('/home/maning/git/shot/data/SSDA_split/multi_shot',domain='clipart',shot=shot)