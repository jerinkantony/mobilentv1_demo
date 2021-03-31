import pyAesCrypt
from cryptography.fernet import Fernet
from scipy.io.wavfile import write,read
import os
import numpy as np 
import random 
from hashlib import sha1

'''
import base64
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

password_provided = "password"  # This is input in the form of a string
password = password_provided.encode()  # Convert to type bytes
salt = b'salt1_'  # CHANGE THIS - recommend using a key from os.urandom(16), must be of type bytes
kdf = PBKDF2HMAC(
    algorithm=hashes.SHA256(),
    length=32,
    salt=salt,
    iterations=100000,
    backend=default_backend()
)
key = base64.urlsafe_b64encode(kdf.derive(password))  # Can only use kdf once
'''

class Cipher:
    def __init__(self):
        #self.password = Fernet.generate_key()
        self.password = b'LnaaxfMOcT_oB4sZVNcfx4W6Uh4H9Mf3v2Joq_9170M='
        self.bufferSize = 64 * 1024
        self.key = self.password# Fernet.generate_key() #this is your "password"
        self.cipher_suite = Fernet(self.key)
        
    def read_audio(self, audiofile):
        '''Reads encrypted and non encrypted files'''
        sr, data = read(audiofile)
        if self.check_encrypted(audiofile):
            data = self.unshuffle_data(data)
        return data, sr
       
    def write_audio(self, data, sr, filename, dstfolder, encrypt=True):
        '''writes audio data in encrypted format to dst folder with encrypted filename'''
        audiofile_enc = self.get_encrypted_filename(filename)
        basepath, filename=os.path.split(audiofile_enc) 
        audiofile_enc = os.path.join(dstfolder,filename)
        data1 = self.shuffle_data(data)
        self.makedir_safe(dstfolder)
        write(audiofile_enc, sr, data1)
        return audiofile_enc
        
    def encrypt_audiofile(self, srcfile, dstfolder=None, delete_srcfile=False):
        '''writes audio file in encrypted format to dst folder with encrypted filename'''
        if dstfolder is None:
            dstfolder, _ = os.path.split(srcfile)
        if not self.check_encrypted(srcfile):
            sr, data = read(srcfile)
            audiofile_enc = self.write_audio(data, sr, srcfile, dstfolder, encrypt=True)
        else:
            audiofile_enc = movefile(srcfile, dstfolder)
        if delete_srcfile==True and srcfile!=audiofile_enc:
            self.removefile(srcfile)
        return audiofile_enc
        
    def check_encrypted(self, path):
        '''checks if file or folder is already encrypted'''
        basepath, filename=os.path.split(path) 
        name, extension = os.path.splitext(filename)
        if '$enc$' in name: #encrypted file
            return True
        else:
            return False
        
    def get_encrypted_filename(self, srcfile):
        '''returns encrypted filename for input filename'''
        basepath, filename=os.path.split(srcfile) 
        #import pdb;pdb.set_trace()
        name, extension = os.path.splitext(filename)
        name1 = self.encrypt_message(name)+'$enc$'
        filename = os.path.join(basepath,name1+extension)
        return filename
        
    def get_decrypted_filename(self,srcfile):
        '''returns decrypted filename for encrypted filename'''
        head, tail = os.path.split(srcfile)
        if not '$enc$' in tail: #encrypted file
            print(srcfile, 'is not encrypted')
            return srcfile
        tail=tail[:-5]
        tail1 = self.decrypt_message(tail)
        folder_enc = os.path.join(head,tail1)
        return folder_enc 
        
    def get_encrypted_foldername(self, srcfolder):
        '''returns encrypted foldername for input foldername'''
        head, tail = os.path.split(srcfolder)
        tail1 = self.encrypt_message(tail)+'$enc$'
        folder_enc = os.path.join(head,tail1)
        return folder_enc
        
    def get_decrypted_foldername(self,srcfolder):
        '''returns decrypted foldername for input encrypted foldername'''
        head, tail = os.path.split(srcfolder)
        if not '$enc$' in tail: #encrypted file
            print(srcfolder, 'is not encrypted')
            return srcfolder
        tail=tail[:-5]
        tail1 = self.decrypt_message(tail)
        folder_enc = os.path.join(head,tail1)
        return folder_enc 
            
    def encrpt_all_files(self, srcroot, delete_srcfiles=False):
        for path, subdirs, files in os.walk(srcroot):
            for name in files:
                srcaudiofile =os.path.join(path, name)
                print (srcaudiofile)
                self.encrypt_audio_file( srcaudiofile,delete_src=False)
                #import pdb;pdb.set_trace()
    
    def encrpt_subfolders(self, srcroot, dstroot, delete_srcfiles=False):
        if dstroot==None:
            dstroot = srcroot
        if not os.path.exists(dstroot):
            os.makedirs(dstroot)
        
        
        for path, subdirs, files in os.walk(srcroot):
            subpath = path.split(srcroot)[1]
            dst_subpath = os.path.join(dstroot,subpath)
            dst_subpath = self.get_encrypted_foldername(dst_subpath)
            for name in files:  
                import pdb;pdb.set_trace()          
                
    
    def encrypt_message(self,message):
        encoded_message = message.encode()
        encrypted_message = self.cipher_suite.encrypt(encoded_message)
        encrypted_message = encrypted_message.decode()
        #print('encrypted_message',encrypted_message)
        return encrypted_message
        
    def decrypt_message(self,message):
        encrypted_message = message.encode()
        decrypted_message = self.cipher_suite.decrypt(encrypted_message)
        decrypted_message = decrypted_message.decode()
        #print('decrypted_message',decrypted_message) 
        return decrypted_message 
        

    def encryptFile(self,fiename):
        pyAesCrypt.encryptFile(fiename, "data.txt.aes", self.password, self.bufferSize)
    
    def decryptFile(self,fiename):
        pyAesCrypt.decryptFile("data.txt.aes", "temp1.wav", self.password, self.bufferSize)
        
    def encryptFolder(self,folder):
        pass
        
    def shuffle_data(self,data):
        order = np.arange( start=0, stop=len(data), step=1 )
        seed = 1221231
        #import pdb;pdb.set_trace()
        random.Random(seed).shuffle(order)
        #print('order1',order)
        return data[order]

    def unshuffle_data(self,data):
        order = np.arange( start=0, stop=len(data), step=1 )
        seed = 1221231
        random.Random(seed).shuffle(order)
        #print('order2',order)
        l_out = np.zeros(len(data), dtype=data.dtype)
        for i, j in enumerate(order):
            l_out[j] = data[i]
        return l_out
        
    def makedir_safe(self,directory):
        if not os.path.exists(directory):
            os.makedirs(directory)  
             
    def movefile(self, srcfile,dstdir):
        if os.path.isfile(srcfile):
            self.makedir_safe(dstdir)
            dstfile = shutil.move(srcfile, dstdir)
            return dstfile
        else:
            return None
            
             
    def removefile(self,filename):
        if os.path.exists(filename):
          os.remove(filename)
        else:
          print("The file does not exist:", filename) 

if __name__=='__main__':
    cipher = Cipher()
    '''
    enc = cipher.encrypt_message('temp')
    print('enc',enc)
    dec = cipher.decrypt_message(enc)
    print('dec',dec)
    '''
    #Read audio (reads both encrypted and non encrypted)
    audiofile = '/home/skycam/audio_classifier/audio_classifier06/ign_utils/temp.wav'
    data1,sr1 = cipher.read_audio(audiofile) #decrypt and read audio file
    print('data1',data1)
    
    #Write audio encrypted
    dstfolder = '/home/skycam/audio_classifier/audio_classifier06/ign_utils/temp'
    audiofile_enc = cipher.write_audio(data1, sr1, audiofile, dstfolder, encrypt=True)
    #print('audiofile_enc',audiofile_enc)
    data2,sr2 = cipher.read_audio(audiofile_enc) 
    print('data2',data2)
    assert (data1==data2).all(), 'mismatch!'
    
    #Encrypt an existing audio file
    dstfolder = '/home/skycam/audio_classifier/audio_classifier06/ign_utils/temp_enc'
    audiofile_enc = cipher.encrypt_audiofile(srcfile=audiofile, dstfolder=dstfolder, delete_srcfile=False)
    data3,sr2 = cipher.read_audio(audiofile_enc) 
    print('data3',data3)
    
    #Get encrypted folder name and reverse
    srcfolder = '/home/skycam/audio_classifier/audio_classifier06/ign_utils/temp'
    encrypted_folder = cipher.get_encrypted_foldername( srcfolder) #to get encrypted folder name
    print('\nencrypted_folder name:',encrypted_folder)
    decrypted_folder = cipher.get_decrypted_foldername( encrypted_folder) #to get decrypted foder name
    print('\ndecrypted_folder name:',decrypted_folder)
    
    '''
    dstroot = '/home/skycam/audio_classifier/audio_classifier06/ign_utils/temp_enc'
    cipher.encrpt_subfolders( srcfolder, dstroot, delete_srcfiles=False)
    '''
    
    
    
    
    
    
        
