"""
Hello there,
    first off you will likely see a lot of typos on this, to paraphrase Zoolander, I don't read good.

    this code is for taking in rockets simulated in FSPro (in-house GTXR sim software) and drawing rockets out using Tkinter GUI.
    This may not be the best way to do this, but it was the only easy one i found to draw basic shapes and have them save "easily".

    how to use:
    first you need to go download ghost scripts and add it to your systems path, this will require a restart. (current links at line: 185)

    next you will need the components, descison_fitness, and events CSV files, saved in the same location as you are running this file of course as per python (unless future you changes that)
    these files should be outputs from FSPro

    lets get into the nitty gritty:
    please be familiar with how Tkiniter Canvas works before you make any changes. its a strange conglomorate of OOP and graph coordinates

    documentation for drawing shape as well as other Tkinter canvas stuff can be found at New Mexico Tech's website
    https://anzeljg.github.io/rin2/book2/2405/docs/tkinter/index.html

    generally speaking i made all the measurments in imperial, becasue I am an AMERICAN engineering student and therefore will behave as an AMERICAN (i use the m2in factor for this)

    there is a "scale_f" varible that can be used to change the size of the rocket drawing on the canvas, so be aware that if you try and read the measure directily without making
      the scale_f = 1, it will look like a huge number that has no meaning.

    there are 3 parts to this program: (4 if you consider the canvas creation)

        -functions: this is where you should write any functions you use to draw the rocket, you will call them from the looping section

        - Data I/O: this is where we use Pandas to read the CSVs and do a bit of cleaning. this is where the counting variable that
            are used in the next varible will be decalred. only events and components will need these since dession_fittness is a one row : one rocket ratio.
            these conunter will need to be changed if more or less rows are add per rocket output
            (as for why there are multi-rowed rockets, ask Dyllon. its something about not having long rows)

        - looping through rockets: the loop takes slices of the huge pandas and makes them smaller, managble pandas using the aformentioned counters.
            the if statement in this loop actully decides if the rocket gets drawn, according the "is fit" in Desison_Fittness
            after that we (including you, if you modify this) pluck the data points we want.
            then we make the canvas. (i try to leave notes as to what things do but its kinda complicated, so read documents or links left in comments)
            finally draw the componetents with the functions, save it and end the loop with a root.destroy() (reason stated next to that command)

    Anyway, that should be all the advise you need to start. read the comments i left and leave helpful ones of your own
-Nick Mazzeo, @Nick M in slack
"""

from tkinter import *

import numpy as np
import math #only need math for the acos function in nose cone
import pandas as pd
import re #regex is used to parse teh file name for the save name (organizational)
verson_name= re.search(r"([vV]\d{1}[\.]*\d*)",f'{__file__}').group()
# print(verson_name)

from PIL import ImageTk, Image
from datetime import datetime
#time and date for save file naming stuff, https://www.programiz.com/python-programming/datetime/current-datetime
now= datetime.now()
dt_string = now.strftime("%Y-%m-%d__%H-%M-%S")
#the following is from https://www.codegrepper.com/code-examples/python/how+to+create+a+directory+and+save+images+in+it+using+os+
import os #purpose is to make a file to deposit stuff into
os.mkdir(f'sim_drawings{verson_name}_{dt_string}')

#variable section:
scale_f= 5
m2in = 39.3701

starting_x=10.0*scale_f #start point a little right for margin purposes
#next_p is the "next starting point", this allows us to keep the starting_x through the program,while still haveing the next posion to start from
next_p=starting_x
####################################################### functions: ##########################################################################################

def body(x_pos,L,colour):
    rocket_canvas.create_rectangle(x_pos,center_y+(odia/2),x_pos+L,center_y-(odia/2),fill=colour)
    r=x_pos+L
    return float(r)


def nose_cone(width,x_pos):
    R=width/2.
    tot_nos_L = 6.0*width

    #haak series von karman curve:
    #T= arccos(1-(2(current length)/tot_nos_L))
    #y=(R/math.sqrt(math.pi))*math.sqrt(T-(math.sin(2T)/2))

    x_list=np.linspace(x_pos,(x_pos+tot_nos_L), num = 25, endpoint= True)
    # print(x_list)
    y_list=[]
    for i in x_list[:-1]:
        T= math.acos(1-((2*(i-starting_x)/tot_nos_L)))
        y=(R/math.sqrt(math.pi))*math.sqrt(T-(math.sin(2*T)/2))
        y_list.append(y)
    y_list.append(R)
    # print(y_list)

    tmpi=0
    tmpj=0
    for i in x_list:
        # print(np.where(x_list == i)[0])
        j= y_list[np.where(x_list == i)[0][0]]
        # print(j)
        rocket_canvas.create_line(tmpi,center_y+tmpj,i,center_y+j,fill="blue")
        rocket_canvas.create_line(tmpi,center_y-tmpj,i,center_y-j,fill="blue")
        tmpi=i
        tmpj=j

    return (x_pos + tot_nos_L)

#fins(fin_dist,root_c,span,tip,"color")
def fins(x_pos,r_chord,y_span,x_tip,sweep,colour):
        #^starting_x
                #^root_c
                        #^span
                                #^tip
    #pyhton functions recieve in the same order they are sent (see line ~390)
    begining_x=x_pos #i made this because the way the thing intitally was odd and i was too lazy to change all its variables
    rad=odia/2

    rocket_canvas.create_polygon((begining_x), (center_y+rad), (begining_x+sweep), (center_y+rad+y_span), (begining_x+sweep+x_tip), (center_y+rad+y_span), \
        (begining_x+r_chord), (center_y+rad) , fill=colour)
    rocket_canvas.create_polygon((begining_x), (center_y-rad), (begining_x+sweep), (center_y-rad-y_span), (begining_x+sweep+x_tip), (center_y-rad-y_span), \
        (begining_x+r_chord), (center_y-rad) , fill=colour)

    # rocket_canvas.create_line(begining_x,center_y+odia , (begining_x-sweep),(center_y+rad) ,fill=colour)

    pass

def motor(motorLength,motorWidth, motorDis, colour): #this function is likley obselete and may be removed
    rocket_canvas.create_rectangle(motorDis,center_y+(motorWidth/2),motorDis+motorLength,center_y-(motorWidth/2),fill=colour)
    pass

#center_of(starting_x,rocket_cg) <--passing parameters
def center_of(nose_t,nose_d):
    #function to add center of graity symbol to diagram, the 2 symbols exspected to be added are
    #for the whole rocket from nose and the propellent center (likely from the nose)
    rocket_canvas.create_oval(nose_t+nose_d-7.5,center_y+7.5,nose_t+nose_d+7.5,center_y-7.5,fill="black")
    rocket_canvas.create_arc(nose_t+nose_d-7.5,center_y+7.5,nose_t+nose_d+7.5,center_y-7.5,fill="white")
    rocket_canvas.create_arc(nose_t+nose_d-7.5,center_y+7.5,nose_t+nose_d+7.5,center_y-7.5,fill="white",start=180,extent=90)

    # #for propellent cg: may or may not add
    # rocket_canvas.create_oval(nose_t+prop_d-7.5,center_y+7.5,nose_t+prop_d+7.5,center_y-7.5,fill="red")
    # rocket_canvas.create_arc(nose_t+prop_d-7.5,center_y+7.5,nose_t+prop_d+7.5,center_y-7.5,fill="white")
    # rocket_canvas.create_arc(nose_t+prop_d-7.5,center_y+7.5,nose_t+prop_d+7.5,center_y-7.5,fill="white",start=180,extent=90)

def scale_bar(scale,d,d2):
    mid=(d+d2)/2
    # font=tkFont.Font(size=10)
    #black background
    rocket_canvas.create_rectangle(starting_x,center_y+d,starting_x+300*scale,center_y+d2,fill="black")
    rocket_canvas.create_text(starting_x,center_y+d2+10,fill="black",text="0",font=('25'))
    #6 inch scale line:
    rocket_canvas.create_rectangle(starting_x,center_y+mid,starting_x+6*scale,center_y+d2,fill="white")
    rocket_canvas.create_text(starting_x+6*scale,center_y+d2+10,fill="black",text="6",font=('25'))
    #12 inch scale line:
    rocket_canvas.create_rectangle(starting_x+6*scale,center_y+d,starting_x+12*scale,center_y+mid,fill="white")
    rocket_canvas.create_text(starting_x+12*scale,center_y+d2+10,fill="black",text="12",font=('25'))
    #24 inch scale line:
    rocket_canvas.create_rectangle(starting_x+12*scale,center_y+mid,starting_x+24*scale,center_y+d2,fill="white")
    rocket_canvas.create_text(starting_x+24*scale,center_y+d2+10,fill="black",text="24",font=('25'))
    #36 inch scale line:
    rocket_canvas.create_rectangle(starting_x+24*scale,center_y+d,starting_x+36*scale,center_y+mid,fill="white")
    rocket_canvas.create_text(starting_x+36*scale,center_y+d2+10,fill="black",text="36",font=('25'))
    #72 inch scale line:
    rocket_canvas.create_rectangle(starting_x+36*scale,center_y+mid,starting_x+72*scale,center_y+d2,fill="white")
    rocket_canvas.create_text(starting_x+72*scale,center_y+d2+10,fill="black",text="72",font=('25'))
    #100 inch scale line:
    rocket_canvas.create_rectangle(starting_x+72*scale,center_y+d,starting_x+100*scale,center_y+mid,fill="white")
    rocket_canvas.create_text(starting_x+100*scale,center_y+d2+10,fill="black",text="100",font=('25'))
    #200 inch line:
    rocket_canvas.create_rectangle(starting_x+100*scale,center_y+mid,starting_x+200*scale,center_y+d2,fill="white")
    rocket_canvas.create_text(starting_x+200*scale,center_y+d2+10,fill="black",text="200",font=('25'))
    #225 inch line:
    rocket_canvas.create_rectangle(starting_x+200*scale,center_y+d,starting_x+225*scale,center_y+mid,fill="white")
    rocket_canvas.create_text(starting_x+225*scale,center_y+d2+10,fill="black",text="225",font=('25'))
    #250 inch line:
    rocket_canvas.create_rectangle(starting_x+225*scale,center_y+mid,starting_x+250*scale,center_y+d2,fill="white")
    rocket_canvas.create_text(starting_x+250*scale,center_y+d2+10,fill="black",text="250",font=('25'))
    #275 inch line:
    rocket_canvas.create_rectangle(starting_x+250*scale,center_y+d,starting_x+275*scale,center_y+mid,fill="white")
    rocket_canvas.create_text(starting_x+275*scale,center_y+d2+10,fill="black",text="275",font=('25'))
    #300 inch line:
    rocket_canvas.create_rectangle(starting_x+275*scale,center_y+mid,starting_x+300*scale,center_y+d2,fill="white")
    rocket_canvas.create_text(starting_x+300*scale,center_y+d2+10,fill="black",text="300",font=('25'))

def sim_label(num,apex,tube,length):
    # try and keep these the same as the ones in the def motor_cross_section
    x1=1000 #booster
    y1=230
    x2=250 #sustatiner
    y2=230
    # sim #
    rocket_canvas.create_text((x1+x2)/2,y1-20,fill="black",text=f"Rocket #{num}",font=('30'))
    # apogee
    rocket_canvas.create_text(x1+500,y1-20,fill="black",text=f"Apogee: {round(apex,4)} km",font=('30'))
    # tube diam
    rocket_canvas.create_text(x1+500,y1,fill="black",text=f"Tube diameter: {tube/scale_f} in",font=('30'))
    #rocket length
    rocket_canvas.create_text(x1+500,y1+20,fill="black",text=f"Rocket length: {round(((length-starting_x)/scale_f),2)} in = {round(((length-starting_x)/(scale_f*m2in)),4)} m",font=('30'))


def motor_cross_section(inMotB, outMotB, inMotS, outMotS):
    #x and y give cords for the propellent diagram
    x1=1000 #booster
    y1=230
    x2=250 #sustatiner
    y2=230

    scale_M=.5 # the scale of .5 is given to the variables becasue of the way the circle is drawn from corner to corner, while still keeping them centered
    indepentednt_scale= 2 # test to see if it makes the scale look bigger, does mean that new scale bar for motors will be needed
    rocket_canvas.create_oval(x1+(scale_M*outMotB*indepentednt_scale),y1+(scale_M*outMotB*indepentednt_scale),x1-(scale_M*outMotB*indepentednt_scale),y1-(scale_M*outMotB*indepentednt_scale), fill='#46BE8F')
    rocket_canvas.create_oval(x1+(scale_M*inMotB*indepentednt_scale),y1+(scale_M*inMotB*indepentednt_scale),x1-(scale_M*inMotB*indepentednt_scale),y1-(scale_M*inMotB*indepentednt_scale), fill='#FF9000')
    rocket_canvas.create_text(x1+120,y1,fill="black",text="<-- Booster",font=('20'))

    rocket_canvas.create_oval(x2+(scale_M*outMotS*indepentednt_scale),y2+(scale_M*outMotS*indepentednt_scale),x2-(scale_M*outMotS*indepentednt_scale),y2-(scale_M*outMotS*indepentednt_scale), fill='#46BE8F')
    rocket_canvas.create_oval(x2+(scale_M*inMotS*indepentednt_scale),y2+(scale_M*inMotS*indepentednt_scale),x2-(scale_M*inMotS*indepentednt_scale),y2-(scale_M*inMotS*indepentednt_scale), fill='#FF9000')
    rocket_canvas.create_text(x2-120,y2,fill="black",text="Sustainer -->",font=('20'))

# ==>IMPORTANT<== links for image save:
"""
https://stackoverflow.com/questions/44587376/oserror-unable-to-locate-ghostscript-on-paths#:~:text=Go%20to%3A%20Control%20Panel%20%2D%3E,your%20ghostscript%20binary%20folder%2C%20e.g.&text=to%20the%20end%20of%20the%20variable.
https://stackoverflow.com/questions/57033158/how-to-save-images-with-the-save-button-on-the-tkinter-in-python
https://stackoverflow.com/questions/9886274/how-can-i-convert-canvas-content-to-an-image
https://www.seniorcare2share.com/how-to-save-canvas-as-image/

I have downloaded ghost scrpts from: https://ghostscript.com/releases/gsdnld.html , as per instructions from https://rnbeads.org/data/installing_rnbeads.html , in order to
get Pillow (PIL) to work here
"""
def save_image():
    rocket_canvas.update() #this update thing is really important, it will make a blank image without it
    rocket_canvas.postscript(file=f"sim_drawings{verson_name}_{dt_string}/rocket.eps",colormode = "color",height=1000, width= 3000)

    img = Image.open(f"sim_drawings{verson_name}_{dt_string}/rocket.eps")
    img.save(f"sim_drawings{verson_name}_{dt_string}/rocket_{i}th.png", "png",dpi=(119.9,119.9))
    # at request of Coung this is code you can use to crop the image down. i think another wat to do this would be to change teh canvas size.
    # #cropping the image, refer to this https://stackoverflow.com/questions/20361444/cropping-an-image-with-python-pillow
    # img = Image.open(f"sim_drawings{verson_name}_{dt_string}/rocket.eps")
    # width, height = img.size
    # right = width - 1000 #change this number if you want to reduce the width
    # bottom = height - 100 #change this number if you want to reduce the height
    # img2 = img.crop((0, 0, right, bottom))
    # img2.save(f"sim_drawings{verson_name}_{dt_string}/rocket_{i}th.png", "png",dpi=(119.9,119.9))
############################################# Data I/O: ###################################################################################################
components = pd.read_csv("components.csv",delimiter=",")
# components.head(12)

#cleaning the data
components = components[components["type"] != "type"]

fitness = pd.read_csv("decision_fitness.csv", delimiter=",",header=0)

event = pd.read_csv("events.csv", delimiter=",")
event = event[event["Event"] != "Event"]


Ccounter=0 #component counter lower range (start of rows with single rocket paramaters)
Ccounter_h=Ccounter+12 #component counter high range (end of rows with single rocket paramaters)
Ecounter=0 #event counter lower range (start of rows with single rocket paramaters)
Ecounter_h=Ecounter+12 #event counter high range (start of rows with single rocket paramaters)

##################################### looping through rockets: ###################################################################################################

"""
there are a few commented for loops  here to make doing limited or revese runs a thing. the top one will be the default case and will do all the rockets first to last,
    it will take a lot of time with big files
"""
# for i in range(0,int(components.shape[0]/12)-1): #may need a -1 subtracted from 2nd range var

for i in range(0,int(50)): #this is a test loop so that you don't need to sit through 100+ rocket creations

# for i in range(int(components.shape[0]/12)-200,int(components.shape[0]/12)-1): #this is to do the last so many rockets, just change the last subtraction in the first range function
    next_p=starting_x
    component_par= components.iloc[Ccounter:Ccounter_h]
    #print(component_par)
    fit_par = fitness.iloc[i]
    #print(fit_par)
    event_par=event.iloc[Ecounter:Ecounter_h]

    Ccounter+=12
    Ccounter_h+=12
    Ecounter+=12
    Ecounter_h+=12
    # print(component_par)

    # print(type(fit_par.loc["is fit"]))
    if fit_par.loc["is fit"] != True: #play with this == sign to make it trigger on False rockets if you don't have True ones
        print( f"###############################{i}th#######################################")

    #     print(event_par)


        odia=fit_par["diam:"]*scale_f
        #name change from root_c to root_c1 (the booster root cord)
        root_c1=fit_par["r2:"]*scale_f #sustainer fin root
        root_c2=fit_par["r1:"]*scale_f #booster fin root
        #name change to span1 from span
        span1= fit_par["sp2:"]*scale_f #sustainer
        span2=fit_par["sp1:"]*scale_f #booster
        #name change from tip to tip1
        tip1= fit_par["t2:"]*scale_f
        tip2= fit_par["t1:"]*scale_f
        #changed sweep to sweep1
        sweep1= fit_par["sw2:"]*scale_f
        sweep2= fit_par["sw1:"]*scale_f
        #apogee (for cool info on picture)
        apo = fit_par["Apogee(km):"]

        """body1 parachute1 parachute2 body2 motor1 body3 body4 parachute3 body5 motor2"""

        #following are lenths of components past from the component files
        bod1=component_par.iloc[1]["l"]*scale_f*m2in
        para1=component_par.iloc[2]["l"]*scale_f*m2in
        coupler1=component_par.iloc[3]["l"]*scale_f*m2in
        para2=component_par.iloc[4]["l"]*scale_f*m2in
        bod2=component_par.iloc[5]["l"]*scale_f*m2in
        motor_sus_L=component_par.iloc[6]["l"]*scale_f*m2in
        bod3=component_par.iloc[7]["l"]*scale_f*m2in
        bod4=component_par.iloc[8]["l"]*scale_f*m2in
        para3=component_par.iloc[9]["l"]*scale_f*m2in
        bod5=component_par.iloc[10]["l"]*scale_f*m2in
        motor_boost_L=component_par.iloc[11]["l"]*scale_f*m2in

        rocket_cg=float(event_par.iloc[0]["cg"])*scale_f*m2in


        fin_loc=[float(x) for x in event_par.iloc[0]["d_finsets"][1:-1].split(",")] #this cleans the string from events "[#,#]" into a list of floats[#,#]
        fin_dist1= -1*fin_loc[0]*scale_f*m2in #sustainer
        fin_dist2= -1*fin_loc[1]*scale_f*m2in #booster


        innerMot_boost = fit_par["dia_1:"]*scale_f
        outerMot_boost = fit_par["odia_1:"]*scale_f
        innerMot_sus = fit_par["dia_2:"]*scale_f
        outerMot_sus = fit_par["odia_2:"]*scale_f

        #print statments for testing:
        # print(f"odia: {odia} \nnose: {odia*6.0}\nroot_c1: {root_c1} \nroot_c2: {root_c2}\nspan1: {span1} \nspan2: {span2}")
        # print(f"sweep1: {sweep1} \nsweep2: {sweep2} \nrocket_cg: {rocket_cg} \nbod1: {bod1}")
        # print(f"para1: {para1}\npara2: {para2}\nbod2: {bod2}\nmotor_sus_L: {motor_sus_L}\nbody3: {bod3}\nbod4: {bod4}")
        # print(f"para3: {para3} \nbod5: {bod5} \nmotor_boost_L: {motor_boost_L}")
        # print(f"fin_dist1: {fin_dist1}\nfin_dist2: {fin_dist2}")
        # print(f"innerMot_boost: {innerMot_boost}\nouterMot_boost: {outerMot_boost}")
        # print(f"rocket_cg: {rocket_cg}")


        # crating the canvas:
        root = Tk()
        root.title('Pretty picture')
        root.geometry("3000x1000")
        # root.pack(fill="both", expand=True)

        maxlength= 3000
        maxwidth= 1000
        center_y=maxwidth/2

        #straight up code from stack overflow: https://stackoverflow.com/questions/7727804/tkinter-using-scrollbars-on-a-canvas
        frame=Frame(root)
        frame.pack(expand=True, fill="both") #.grid(row=0,column=0)
        # canvas=Canvas(frame,bg='#FFFFFF',width=300,height=300,scrollregion=(0,0,500,500))

        #creation of rocket_canvas object
        # for the time being the background (bg) will be this shade of piss yellow (#F3DB00) or depressing orange (#F3AA00), rather than white so i stop flash banging myself everytime i run this at night
        rocket_canvas = Canvas(frame, bg="#F3AA00", height=maxwidth, width=maxlength,scrollregion=(0,0,maxlength,maxwidth))


        hbar=Scrollbar(frame,orient=HORIZONTAL)
        hbar.pack(side=BOTTOM,fill=X)
        hbar.config(command=rocket_canvas.xview)
        vbar=Scrollbar(frame,orient=VERTICAL)
        vbar.pack(side=RIGHT,fill=Y)
        vbar.config(command=rocket_canvas.yview)
        # canvas.config(width=300,height=300)
        rocket_canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        rocket_canvas.pack(side=LEFT,expand=True,fill="both")

        #scale bar
        scale_bar(scale_f,100,100+50)
        #motor cross section
        motor_cross_section(innerMot_boost, outerMot_boost, innerMot_sus, outerMot_sus)

        """body1 parachute1 parachute2 body2 motor1 body3 body4 parachute3 body5 motor2    """

        #nosecone
        next_p=nose_cone(odia,next_p)
        #body1
        next_p=body(next_p,bod1,"#E2F3DC") #color is light green
        #para1
        next_p=body(next_p,para1,"#9F5FCF") #color is light purple
        #coupler1
        next_p=body(next_p,coupler1,"#7131A1") #color is dark purple
        #para2
        next_p=body(next_p,para2,"#9F5FCF") #color s light purple
        #bod2
        next_p=body(next_p,bod2,"#E2F3DC") #color is light green
        #motor1 (sustainer)
        next_p=body(next_p,motor_sus_L,"#FFABAB") #color is pink
        #bod3
        next_p=body(next_p,bod3,"#E2F3DC") #color is light green
        #bod4
        next_p=body(next_p,bod4,"#E2F3DC") #color is light green
        #para3
        next_p=body(next_p,para3,"#9F5FCF") #color is light purple
        #bod5
        next_p=body(next_p,bod5,"#E2F3DC") #color is light green
        #motor2 (booster boi)
        next_p=body(next_p,motor_boost_L,"#FFABAB") #color is pink
        #fin1 (sustainer)
        # fins(starting_x+ rocket_cg+ fin_dist1,root_c1,span1,tip1,sweep1,"#5FA6DA") #color is light blue
        fins(starting_x+ fin_dist1+ rocket_cg,root_c1,span1,tip1,sweep1,"#5FA6DA") #color is light blue
        #fin2 (booster)
        fins(starting_x+ fin_dist2+ rocket_cg,root_c2,span2,tip2,sweep2,"#3990CE") #color is light blue
        # fins(starting_x+ rocket_cg+ fin_dist2-root_c1-root_c2,root_c2,span2,tip2,sweep2,"purple")

        #print statments for testing:
        # print(next_p)
        print ("sus:")
        print(fin_dist1/scale_f)
        print(( rocket_cg+ fin_dist1)/scale_f)
        print("\nboost:")
        print(fin_dist2/scale_f)
        print(( rocket_cg+ fin_dist2)/scale_f)
        # print(starting_x+ rocket_cg+ fin_dist2-root_c1-root_c2)
        # print(starting_x+ rocket_cg+ fin_dist2-next_p)
        # print(f"{rocket_cg/scale_f} \n {fin_loc} \n{fin_dist1/scale_f} , {fin_dist2/scale_f}")
        # #motor
        # # motor(mlength,mwidth,mdis,"red")
        # #cg markers

        # simulation labels
        sim_label(i,apo,odia,next_p)
        center_of(starting_x,rocket_cg)

        # this shows the center line we are building from
        rocket_canvas.create_line(0,center_y,maxlength,center_y,dash=(10,20),fill="green")

        save_image()

        root.destroy() #command to close window, so you don't end up with thousands of windows open

######################################################################################################################################################


