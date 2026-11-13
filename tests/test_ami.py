import numpy as np
from camix_director.ami import camera_mapping,word_intervals,active

def test_mapping_is_not_assumed_from_camera_number():
    html='<tr><td>ES2002a</td><td>PM,Closeup4,1, B; ID,Closeup1,0, A;</td></tr>'
    mapping=camera_mapping(html,'ES2002a')
    assert mapping['B']==(4,1) and mapping['A']==(1,0)

def test_overlap_can_be_left_unlabelled():
    a=active([(0,1)],np.array([.2,.7,1.1]));b=active([(.5,1.5)],np.array([.2,.7,1.1]))
    assert (a.astype(int)+b.astype(int)).tolist()==[1,2,1]

def test_other_speakers_and_four_person_overlap_are_not_silent_keep():
    from camix_director.ami import transcript_preferences
    class Archive:
        def read(self,path):
            agent=path.split('.')[1]
            spans={'A':[(0,1)],'B':[(1,2)],'C':[(.5,1.5),(2,3)],'D':[]}[agent]
            return ('<root>'+''.join(f'<w starttime="{a}" endtime="{b}">word</w>' for a,b in spans)+'</root>').encode()
    times=np.array([.2,.7,1.2,1.7,2.2,3.1])
    prefs=transcript_preferences(Archive(),'TEST',0,times,np.ones((6,2)),np.ones((6,2,2)),2)
    assert prefs[0].tolist()==[1,1,0] and prefs[3].tolist()==[0,0,1]
    assert np.isnan(prefs[[1,2,4,5]]).all()
