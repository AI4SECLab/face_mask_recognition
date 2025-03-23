from django.shortcuts import render,redirect
from .forms import usernameForm,DateForm,UsernameAndDateForm, DateForm_2
from django.contrib import messages
from django.contrib.auth.models import User
import cv2
import time
from attendance_system_facial_recognition.settings import BASE_DIR
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
from django.contrib.auth.decorators import login_required
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import datetime
from django_pandas.io import read_frame
from users.models import Present, Time
import seaborn as sns
import pandas as pd
from django.db.models import Count
#import mpld3
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from matplotlib import rcParams
import math
import sys
sys.path.append("../")
from face_dataset import create_dataset
mpl.use('Agg')


#utility functions:
def username_present(username):
	if User.objects.filter(username=username).exists():
		return True
	
	return False

# def create_dataset(username):
# 	id = username
# 	if(os.path.exists('face_recognition_data/training_dataset/{}/'.format(id))==False):
# 		os.makedirs('face_recognition_data/training_dataset/{}/'.format(id))
# 	directory='face_recognition_data/training_dataset/{}/'.format(id)

# 	# Detect face
# 	#Loading the HOG face detector and the shape predictpr for allignment

# 	print("[INFO] Loading the facial detector")
# 	detector = dlib.get_frontal_face_detector()
# 	predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')   #Add path to the shape predictor ######CHANGE TO RELATIVE PATH LATER
# 	fa = FaceAligner(predictor , desiredFaceWidth = 96)
# 	#capture images from the webcam and process and detect the face
# 	# Initialize the video stream
# 	print("[INFO] Initializing Video stream")
# 	vs = VideoStream(src=0).start()
# 	#time.sleep(2.0) ####CHECK######

# 	# Our identifier
# 	# We will put the id here and we will store the id with a face, so that later we can identify whose face it is
	
# 	# Our dataset naming counter
# 	sampleNum = 0
# 	# Capturing the faces one by one and detect the faces and showing it on the window
# 	while(True):
# 		# Capturing the image
# 		#vs.read each frame
# 		frame = vs.read()
# 		#Resize each image
# 		frame = imutils.resize(frame ,width = 800)
# 		#the returned img is a colored image but for the classifier to work we need a greyscale image
# 		#to convert
# 		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 		#To store the faces
# 		#This will detect all the images in the current frame, and it will return the coordinates of the faces
# 		#Takes in image and some other parameter for accurate result
# 		faces = detector(gray_frame,0)
# 		#In above 'faces' variable there can be multiple faces so we have to get each and every face and draw a rectangle around it.
# 		for face in faces:
# 			print("inside for loop")
# 			(x,y,w,h) = face_utils.rect_to_bb(face)
# 			face_aligned = fa.align(frame,gray_frame,face)
# 			# Whenever the program captures the face, we will write that is a folder
# 			# Before capturing the face, we need to tell the script whose face it is
# 			# For that we will need an identifier, here we call it id
# 			# So now we captured a face, we need to write it in a file
# 			sampleNum = sampleNum+1
# 			# Saving the image dataset, but only the face part, cropping the rest
# 			if face is None:
# 				print("face is none")
# 				continue
# 			cv2.imwrite(directory+'/'+str(sampleNum)+'.jpg'	, face_aligned)
# 			face_aligned = imutils.resize(face_aligned ,width = 400)
# 			#cv2.imshow("Image Captured",face_aligned)
# 			# @params the initial point of the rectangle will be x,y and
# 			# @params end point will be x+width and y+height
# 			# @params along with color of the rectangle
# 			# @params thickness of the rectangle
# 			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
# 			# Before continuing to the next loop, I want to give it a little pause
# 			# waitKey of 100 millisecond
# 			cv2.waitKey(50)
# 		#Showing the image in another window
# 		#Creates a window with window name "Face" and with the image img
# 		cv2.imshow("Add Images",frame)
# 		#Before closing it we need to give a wait command, otherwise the open cv wont work
# 		# @params with the millisecond of delay 1
# 		cv2.waitKey(1)
# 		#To get out of the loop
# 		if(sampleNum>300):
# 			break
# 	#Stoping the videostream
# 	vs.stop()
# 	# destroying all the windows
# 	cv2.destroyAllWindows()


def predict(face_aligned,svc,threshold=0.7):
	face_encodings=np.zeros((1,128))
	try:
		x_face_locations=face_recognition.face_locations(face_aligned)
		faces_encodings=face_recognition.face_encodings(face_aligned,known_face_locations=x_face_locations)
		if(len(faces_encodings)==0):
			return ([-1],[0])

	except:

		return ([-1],[0])

	prob=svc.predict_proba(faces_encodings)
	result=np.where(prob[0]==np.amax(prob[0]))
	if(prob[0][result[0]]<=threshold):
		return ([-1],prob[0][result[0]])

	return (result[0],prob[0][result[0]])


def vizualize_Data(embedded, targets,):
	X_embedded = TSNE(n_components=2).fit_transform(embedded)
	for i, t in enumerate(set(targets)):
		idx = targets == t
		plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)
	plt.legend(bbox_to_anchor=(1, 1));
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()	
	plt.savefig('./recognition/static/recognition/img/training_visualisation.png')
	plt.close()



def update_attendance_in_db_in(present, log_time):
	today=datetime.date.today()
	for person in present:
		user=User.objects.get(username=person)
		try:
			qs=Present.objects.get(user=user,date=today)
		except :
			qs= None
		if qs is None:
			if present[person]==True:
				a=Present(user=user,date=today,present=True)
				a.save()
			else:
				a=Present(user=user,date=today,present=False)
				a.save()
		else:
			if present[person]==True:
				qs.present=True
				qs.save(update_fields=['present'])
		if present[person]==True:
			# Get image filename from log_time
			image_filename = log_time[person]['image'] if person in log_time else None
			a=Time(user=user,date=today,time=log_time[person]['time'], out=False, image=image_filename)
			a.save()

def update_attendance_in_db_out(present, log_time):
	today=datetime.date.today()
	for person in present:
		user=User.objects.get(username=person)
		if present[person]==True:
			# Get image filename from log_time
			image_filename = log_time[person]['image'] if person in log_time else None
			a=Time(user=user,date=today,time=log_time[person]['time'], out=True, image=image_filename)
			a.save()

def check_validity_times(times_all):

	if(len(times_all)>0):
		sign=times_all.first().out
	else:
		sign=True
	break_hourss=0
	if(sign==True):
			check=False
			break_hourss=0
			return (check,break_hourss)
	prev=True
	prev_time=times_all.first().time

	for obj in times_all:
		curr=obj.out
		if(curr==prev):
			check=False
			break_hourss=0
			return (check,break_hourss)
		if(curr==False):
			curr_time=obj.time
			to=curr_time
			ti=prev_time
			break_time=((to-ti).total_seconds())/3600
			break_hourss+=break_time
		else:
			prev_time=obj.time
		prev=curr
	return (True,break_hourss)


def convert_hours_to_hours_mins(hours):
	
	h=int(hours)
	hours-=h
	m=hours*60
	m=math.ceil(m)
	return str(str(h)+ " hrs " + str(m) + "  mins")

		

#used
def hours_vs_date_given_employee(present_qs, time_qs, admin=True):
	register_matplotlib_converters()
	df_hours = []
	df_break_hours = []
	qs = present_qs

	for obj in qs:
		date = obj.date
		times_in = time_qs.filter(date=date).filter(out=False).order_by('time')
		times_out = time_qs.filter(date=date).filter(out=True).order_by('time')
		times_all = time_qs.filter(date=date).order_by('time')
		
		# Initialize basic attributes
		obj.time_in = None 
		obj.time_out = None
		obj.hours = 0
		obj.break_hours = 0
		obj.image = None  # Initialize image attribute
		
		# Get first time record for the day to get image
		if (len(times_in) > 0):
			obj.image = times_in.first().image
		
		if (len(times_in) > 0):
			obj.time_in = times_in.first().time
		if (len(times_out) > 0):
			obj.time_out = times_out.last().time
			
		# Calculate hours
		if(obj.time_in is not None and obj.time_out is not None):
			ti = obj.time_in
			to = obj.time_out
			hours = ((to-ti).total_seconds())/3600
			obj.hours = hours
		
		# Calculate break hours  
		(check, break_hourss) = check_validity_times(times_all)
		obj.break_hours = break_hourss if check else 0
		
		# Add to dataframe lists
		df_hours.append(obj.hours)
		df_break_hours.append(obj.break_hours)
		
		# Convert hours to readable format
		obj.hours = convert_hours_to_hours_mins(obj.hours)
		obj.break_hours = convert_hours_to_hours_mins(obj.break_hours)

	# Create dataframe
	df = read_frame(qs)
	df["hours"] = df_hours
	df["break_hours"] = df_break_hours

	# Plot graph
	sns.barplot(data=df, x='date', y='hours')
	plt.xticks(rotation='vertical')
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()
	
	if(admin):
		plt.savefig('./recognition/static/recognition/img/attendance_graphs/hours_vs_date/1.png')
	else:
		plt.savefig('./recognition/static/recognition/img/attendance_graphs/employee_login/1.png')
	plt.close()

	return qs

#used
def hours_vs_employee_given_date(present_qs,time_qs):
	register_matplotlib_converters()
	df_hours=[]
	df_break_hours=[]
	df_username=[]
	qs=present_qs

	for obj in qs:
		user=obj.user
		times_in=time_qs.filter(user=user).filter(out=False)
		times_out=time_qs.filter(user=user).filter(out=True)
		times_all=time_qs.filter(user=user)
		obj.time_in=None
		obj.time_out=None
		obj.hours=0
		obj.hours=0
		obj.image = None  # Initialize image attribute
		# Get first time record for the day to get image
		if (len(times_in) > 0):
			obj.image = times_in.first().image
		if (len(times_in)>0):			
			obj.time_in=times_in.first().time
		if (len(times_out)>0):
			obj.time_out=times_out.last().time
		if(obj.time_in is not None and obj.time_out is not None):
			ti=obj.time_in
			to=obj.time_out
			hours=((to-ti).total_seconds())/3600
			obj.hours=hours
		else:
			obj.hours=0
		(check,break_hourss)= check_validity_times(times_all)
		if check:
			obj.break_hours=break_hourss


		else:
			obj.break_hours=0

		
		df_hours.append(obj.hours)
		df_username.append(user.username)
		df_break_hours.append(obj.break_hours)
		obj.hours=convert_hours_to_hours_mins(obj.hours)
		obj.break_hours=convert_hours_to_hours_mins(obj.break_hours)

	



	df = read_frame(qs)	
	df['hours']=df_hours
	df['username']=df_username
	df["break_hours"]=df_break_hours


	sns.barplot(data=df,x='username',y='hours')
	plt.xticks(rotation='vertical')
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()
	plt.savefig('./recognition/static/recognition/img/attendance_graphs/hours_vs_employee/1.png')
	plt.close()
	return qs


def total_number_employees():
	qs=User.objects.all()
	return (len(qs) -1)
	# -1 to account for admin 



def employees_present_today():
	today=datetime.date.today()
	qs=Present.objects.filter(date=today).filter(present=True)
	return len(qs)




#used	
def this_week_emp_count_vs_date():
	today=datetime.date.today()
	some_day_last_week=today-datetime.timedelta(days=7)
	monday_of_last_week=some_day_last_week-  datetime.timedelta(days=(some_day_last_week.isocalendar()[2] - 1))
	monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
	qs=Present.objects.filter(date__gte=monday_of_this_week).filter(date__lte=today)
	str_dates=[]
	emp_count=[]
	str_dates_all=[]
	emp_cnt_all=[]
	cnt=0
	
	



	for obj in qs:
		date=obj.date
		str_dates.append(str(date))
		qs=Present.objects.filter(date=date).filter(present=True)
		emp_count.append(len(qs))


	while(cnt<5):

		date=str(monday_of_this_week+datetime.timedelta(days=cnt))
		cnt+=1
		str_dates_all.append(date)
		if(str_dates.count(date))>0:
			idx=str_dates.index(date)

			emp_cnt_all.append(emp_count[idx])
		else:
			emp_cnt_all.append(0)

	
	
	



	df=pd.DataFrame()
	df["Ngày"]=str_dates_all
	df["Số lượng người dùng"]=emp_cnt_all
	
	
	sns.lineplot(data=df,x='Ngày',y='Số lượng người dùng')
	plt.savefig('./recognition/static/recognition/img/attendance_graphs/this_week/1.png')
	plt.close()

#used
def last_week_emp_count_vs_date():
	today=datetime.date.today()
	some_day_last_week=today-datetime.timedelta(days=7)
	monday_of_last_week=some_day_last_week-  datetime.timedelta(days=(some_day_last_week.isocalendar()[2] - 1))
	monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
	qs=Present.objects.filter(date__gte=monday_of_last_week).filter(date__lt=monday_of_this_week)
	str_dates=[]
	emp_count=[]
	str_dates_all=[]
	emp_cnt_all=[]
	cnt=0
	for obj in qs:
		date=obj.date
		str_dates.append(str(date))
		qs=Present.objects.filter(date=date).filter(present=True)
		emp_count.append(len(qs))
	while(cnt<5):
		date=str(monday_of_last_week+datetime.timedelta(days=cnt))
		cnt+=1
		str_dates_all.append(date)
		if(str_dates.count(date))>0:
			idx=str_dates.index(date)
			emp_cnt_all.append(emp_count[idx])
			
		else:
			emp_cnt_all.append(0)
	df=pd.DataFrame()
	df["Ngày"]=str_dates_all
	df["Số lượng người dùng"]=emp_cnt_all
	sns.lineplot(data=df,x='Ngày',y='Số lượng người dùng')
	plt.savefig('./recognition/static/recognition/img/attendance_graphs/last_week/1.png')
	plt.close()

# Create your views here.
def home(request):
	return render(request, 'recognition/home.html')

@login_required
def dashboard(request):
	if(request.user.username=='admin'):
		print("admin")
		return render(request, 'recognition/admin_dashboard.html')
	else:
		print("not admin")
		return render(request,'recognition/employee_dashboard.html')

@login_required
def add_photos(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	if request.method=='POST':
		form=usernameForm(request.POST)
		data = request.POST.copy()
		username=data.get('username')
		if username_present(username):
			create_dataset(username)
			messages.success(request, f'Đã thêm thành công')
			return redirect('add-photos')
		else:
			messages.warning(request, f'Không có người dùng. Hãy đăng ký người dùng trước.')
			return redirect('dashboard')
	else:
		form=usernameForm()
		return render(request,'recognition/add_photos.html', {'form' : form})


from mask_detection import mask_detection
def mark_your_attendance(request):
	save_path = r".\demo.avi"
	present, log_time = mask_detection(is_2_write=False, save_path=save_path)
	update_attendance_in_db_in(present, log_time)
	return redirect('home')


def mark_your_attendance_out(request):
	save_path = r".\demo.avi"
	present, log_time = mask_detection(is_2_write=False, save_path=save_path)
	update_attendance_in_db_out(present, log_time)
	return redirect('home')

from mask_train import __train

@login_required
def train(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	# training_dir='face_recognition_data/training_dataset'
	# count=0
	# for person_name in os.listdir(training_dir):
	# 	curr_directory=os.path.join(training_dir,person_name)
	# 	if not os.path.isdir(curr_directory):
	# 		continue
	# 	for imagefile in image_files_in_folder(curr_directory):
	# 		count+=1
	# X=[]
	# y=[]
	# i=0
	# for person_name in os.listdir(training_dir):
	# 	print(str(person_name))
	# 	curr_directory=os.path.join(training_dir,person_name)
	# 	if not os.path.isdir(curr_directory):
	# 		continue
	# 	for imagefile in image_files_in_folder(curr_directory):
	# 		print(str(imagefile))
	# 		image=cv2.imread(imagefile)
	# 		try:
	# 			X.append((face_recognition.face_encodings(image)[0]).tolist())
	# 			y.append(person_name)
	# 			i+=1
	# 		except:
	# 			print("removed")
	# 			os.remove(imagefile)
	
	# targets=np.array(y)
	# encoder = LabelEncoder()
	# encoder.fit(y)
	# y=encoder.transform(y)
	# X1=np.array(X)
	# print("shape: "+ str(X1.shape))
	# np.save('face_recognition_data/classes.npy', encoder.classes_)
	# svc = SVC(kernel='linear',probability=True)
	# svc.fit(X1,y)
	# svc_save_path="face_recognition_data/svc.sav"
	# with open(svc_save_path, 'wb') as f:
	# 	pickle.dump(svc,f)
	# vizualize_Data(X1,targets)
	node_dict = {'input':'data_1:0',
                'detection_bboxes':'loc_branch_concat_1/concat:0',
                'detection_scores':'cls_branch_concat_1/concat:0'}
	__train("../MaskTheFace/data", "../face_mask_detection.pb", "../MaskTheFace/data_masked", node_dict)
	messages.success(request, f'Training Complete.')
	return render(request,"recognition/train.html")

@login_required
def not_authorised(request):
	return render(request,'recognition/not_authorised.html')



@login_required
def view_attendance_home(request):
	total_num_of_emp=total_number_employees()
	emp_present_today=employees_present_today()
	this_week_emp_count_vs_date()
	last_week_emp_count_vs_date()
	return render(request,"recognition/view_attendance_home.html", {'total_num_of_emp' : total_num_of_emp, 'emp_present_today': emp_present_today})


@login_required
def view_attendance_date(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	qs=None
	time_qs=None
	present_qs=None
	if request.method=='POST':
		form=DateForm(request.POST)
		if form.is_valid():
			date=form.cleaned_data.get('date')
			print("date:"+ str(date))
			time_qs=Time.objects.filter(date=date)
			present_qs=Present.objects.filter(date=date)
			if(len(time_qs)>0 or len(present_qs)>0):
				qs=hours_vs_employee_given_date(present_qs,time_qs)
				# Add image paths to queryset
				# for item in qs:
				# 	time_records = Time.objects.filter(user=item.user, date=date)
				# 	if time_records.exists() and time_records.first().image:
				# 		item.image = time_records.first().image
				return render(request,'recognition/view_attendance_date.html', {'form' : form,'qs' : qs })
			else:
				messages.warning(request, f'No records for selected date.')
				return redirect('view-attendance-date')
	else:
			form=DateForm()
			return render(request,'recognition/view_attendance_date.html', {'form' : form, 'qs' : qs})


@login_required
def view_attendance_employee(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	time_qs=None
	present_qs=None
	qs=None

	if request.method=='POST':
		form=UsernameAndDateForm(request.POST)
		if form.is_valid():
			username=form.cleaned_data.get('username')
			if username_present(username):
				
				u=User.objects.get(username=username)
				
				time_qs=Time.objects.filter(user=u)
				present_qs=Present.objects.filter(user=u)
				date_from=form.cleaned_data.get('date_from')
				date_to=form.cleaned_data.get('date_to')
				
				if date_to < date_from:
					messages.warning(request, f'Invalid date selection.')
					return redirect('view-attendance-employee')
				else:
					time_qs=time_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
					present_qs=present_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
					if (len(time_qs)>0 or len(present_qs)>0):
						qs = hours_vs_date_given_employee(present_qs,time_qs,admin=True)
						for item in qs:
							print(item.image)
						return render(request,'recognition/view_attendance_employee.html', {'form' : form, 'qs' :qs})
					else:
						messages.warning(request, f'No records for selected duration.')
						return redirect('view-attendance-employee')
			else:
				print("invalid username")
				messages.warning(request, f'No such username found.')
				return redirect('view-attendance-employee')
	else:
			form=UsernameAndDateForm()
			return render(request,'recognition/view_attendance_employee.html', {'form' : form, 'qs' :qs})
@login_required
def view_my_attendance_employee_login(request):
	if request.user.username=='admin':
		return redirect('not-authorised')
	qs=None
	time_qs=None
	present_qs=None
	if request.method=='POST':
		form=DateForm_2(request.POST)
		if form.is_valid():
			u=request.user
			time_qs=Time.objects.filter(user=u)
			present_qs=Present.objects.filter(user=u)
			date_from=form.cleaned_data.get('date_from')
			date_to=form.cleaned_data.get('date_to')
			if date_to < date_from:
					messages.warning(request, f'Invalid date selection.')
					return redirect('view-my-attendance-employee-login')
			else:
					time_qs=time_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
					present_qs=present_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
				
					if (len(time_qs)>0 or len(present_qs)>0):
						qs=hours_vs_date_given_employee(present_qs,time_qs,admin=False)
						return render(request,'recognition/view_my_attendance_employee_login.html', {'form' : form, 'qs' :qs})
					else:
						
						messages.warning(request, f'No records for selected duration.')
						return redirect('view-my-attendance-employee-login')
	else:
			form=DateForm_2()
			return render(request,'recognition/view_my_attendance_employee_login.html', {'form' : form, 'qs' :qs})