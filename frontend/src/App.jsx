import { useEffect, useRef, useState } from 'react';
import { api } from './api';

const ACCEPTED_AUDIO = '.wav,.m4a,.mp3,.webm,audio/wav,audio/mpeg,audio/mp4,audio/webm';
const STUDY_LEAD = 'Dinesh Kumar Prajapati';

function StudyScreen({ onJoin, onSignIn }) {
  return <main className="study-shell">
    <header className="study-header"><span className="study-mark">VAS</span><span>Voice-verification case study · Part 2</span></header>
    <section className="study-hero">
      <p className="eyebrow">Participant invitation</p>
      <h1>Your voice, two speaking styles.</h1>
      <p className="study-lede">Help us explore whether natural free speech or carefully read pangrams creates more reliable voice-verification signals.</p>
      <div className="study-actions"><button onClick={onJoin}>Take part in the study</button><button className="secondary" onClick={onSignIn}>I already have an account</button></div>
      <p className="study-time">About 8–12 minutes · microphone required · voluntary</p>
    </section>
    <section className="study-grid" aria-label="Study details">
      <article className="study-card"><span className="card-number">01</span><h2>The question</h2><p>Does a speaker-verification system work better when someone speaks naturally, or when they read a phonetically rich sentence?</p></article>
      <article className="study-card"><span className="card-number">02</span><h2>What you’ll do</h2><p>Record 10 short free-speech clips, then read 10 displayed pangrams aloud. You can stop at any time before finishing.</p></article>
      <article className="study-card"><span className="card-number">03</span><h2>How it helps</h2><p>We will compare verification scores across both conditions to understand which enrollment style is more robust.</p></article>
    </section>
    <section className="study-notice">
      <div><p className="eyebrow">Before you begin</p><h2>Participation and privacy</h2></div>
      <div className="study-notice-copy"><p>This case study is led by {STUDY_LEAD} and continues research presented at ICICC 2020 in Springer’s Advances in Intelligent Systems and Computing (AISC) series. If the findings are relevant, we may pursue publication as Part 2: <em>Voice Authentication System — Free Speech vs Pangrams</em>.</p><p><a className="paper-link" href="https://link.springer.com/chapter/10.1007/978-981-15-5113-0_45" target="_blank" rel="noreferrer">View the ICICC 2020 Springer publication ↗</a></p><p>Participation is optional. Please record only your own voice and avoid sharing sensitive or identifying details while speaking freely. Your recordings and derived voice features are used only for this project’s voice-verification analysis.</p></div>
    </section>
    <footer className="study-footer">Voice Access System · Free Speech vs Pangrams case study</footer>
  </main>;
}

function Result({ result }) {
  return result && <div className={`result ${result.type}`} role="status">{result.message}</div>;
}

function AudioInput({ onChange, file, recording, onRecord }) {
  return <div className="audio-input">
    <input id="audio-file" type="file" accept={ACCEPTED_AUDIO} onChange={(event) => onChange(event.target.files?.[0] || null)} />
    <label htmlFor="audio-file">{file ? file.name : 'Choose an audio file'}</label>
    <button className={recording ? 'recording' : 'secondary'} type="button" onClick={onRecord}>
      {recording ? 'Stop recording' : 'Record a sample'}
    </button>
  </div>;
}

function AuthenticationScreen({ token, onLogout, completed = false }) {
  const [file, setFile] = useState(null);
  const [action, setAction] = useState('');
  const [result, setResult] = useState(null);
  const recorder = useRef(null);
  const chunks = useRef([]);

  useEffect(() => () => recorder.current?.stream?.getTracks().forEach((track) => track.stop()), []);

  async function toggleRecording() {
    if (recorder.current?.state === 'recording') return recorder.current.stop();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      chunks.current = [];
      mediaRecorder.ondataavailable = ({ data }) => chunks.current.push(data);
      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks.current, { type: mediaRecorder.mimeType || 'audio/webm' });
        setFile(new File([blob], 'voice-authentication.webm', { type: blob.type }));
        stream.getTracks().forEach((track) => track.stop()); recorder.current = null; setAction('');
      };
      recorder.current = mediaRecorder; mediaRecorder.start(); setResult(null); setAction('recording');
    } catch { setResult({ type: 'error', message: 'Microphone access was unavailable. You can upload an audio file instead.' }); }
  }

  async function authenticate() {
    if (!file) return setResult({ type: 'error', message: 'Record or choose an audio clip to test your voice.' });
    setAction('authenticate'); setResult(null);
    try {
      const response = await api.authenticate(file, token);
      setResult({ type: 'success', message: `Voice verified! Similarity score: ${response.similarity_score} (required: ${response.threshold}).` });
    } catch (error) { setResult({ type: 'error', message: error.message }); } finally { setAction(''); }
  }

  const recording = action === 'recording';
  return <>
    <div className="signed-in"><span><i /> Voice test</span><button className="text-button" onClick={onLogout}>Sign out</button></div>
    <div className="volunteer-intro"><p className="eyebrow">{completed ? 'Enrollment complete' : 'Authentication check'}</p><h2>{completed ? 'Thank you for completing your voice enrollment!' : 'Test your enrolled voice.'}</h2><p>{completed ? 'You can now verify your enrolled voice with a new recording.' : 'Record a fresh short phrase or upload a new clip to verify your voice.'}</p></div>
    <AudioInput file={file} onChange={setFile} recording={recording} onRecord={toggleRecording} />
    {recording && <p className="recording-note">Recording in progress… Say a short phrase, then stop.</p>}
    <button className="enroll-button" disabled={!!action} onClick={authenticate}>{action === 'authenticate' ? 'Checking your voice…' : 'Test my voice'}</button>
    <p className="note">WAV, M4A, MP3, and WebM files are supported.</p>
    <Result result={result} />
  </>;
}

function VolunteerFlow({ token, onLogout }) {
  const [progress, setProgress] = useState(null);
  const [pangram, setPangram] = useState(null);
  const [pangramLoading, setPangramLoading] = useState(false);
  const [file, setFile] = useState(null);
  const [action, setAction] = useState('');
  const [result, setResult] = useState(null);
  const recorder = useRef(null);
  const chunks = useRef([]);

  async function loadPangram() {
    setPangramLoading(true);
    try {
      const response = await api.pangrams(token);
      setPangram(response.pangrams?.[0] || null);
      if (!response.pangrams?.length) announce('error', 'There are no more pangrams available for this session.');
    } catch (error) { announce('error', error.message); } finally { setPangramLoading(false); }
  }

  useEffect(() => {
    api.enrollmentStatus(token).then((status) => {
      setProgress(status);
      const freeSpeechCollected = status.free_speech_collected ?? status.samples_collected ?? 0;
      const freeSpeechTarget = status.free_speech_target ?? 10;
      if (!status.is_complete && freeSpeechCollected >= freeSpeechTarget) loadPangram();
    }).catch((error) => setResult({ type: 'error', message: error.message }));
    return () => recorder.current?.stream?.getTracks().forEach((track) => track.stop());
  }, [token]);

  function announce(type, message) { setResult({ type, message }); }

  async function enroll() {
    if (!file) return announce('error', 'Record or choose one voice sample first.');
    const freeSpeechCollected = progress.free_speech_collected ?? progress.samples_collected ?? 0;
    const freeSpeechTarget = progress.free_speech_target ?? 10;
    const promptType = freeSpeechCollected >= freeSpeechTarget ? 'pangram' : 'free_speech';
    if (promptType === 'pangram' && !pangram) return announce('error', 'Wait for a pangram before recording this sample.');
    setAction('enroll'); setResult(null);
    try {
      const response = await api.enroll(file, token, { promptType, pangramId: pangram?.id });
      setProgress((current) => ({ ...current, ...response }));
      setFile(null);
      announce('success', promptType === 'pangram' ? 'Pangram recording saved. Great reading!' : 'Free-speech recording saved. Thank you!');
      const freeSpeechComplete = response.free_speech_collected >= response.free_speech_target;
      if (!response.is_complete && (promptType === 'pangram' || freeSpeechComplete)) loadPangram();
    } catch (error) { announce('error', error.message); } finally { setAction(''); }
  }

  async function toggleRecording() {
    if (recorder.current?.state === 'recording') return recorder.current.stop();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      chunks.current = [];
      mediaRecorder.ondataavailable = ({ data }) => chunks.current.push(data);
      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks.current, { type: mediaRecorder.mimeType || 'audio/webm' });
        setFile(new File([blob], 'voice-recording.webm', { type: blob.type }));
        stream.getTracks().forEach((track) => track.stop()); recorder.current = null; setAction('');
      };
      recorder.current = mediaRecorder; mediaRecorder.start(); setResult(null); setAction('recording');
    } catch { announce('error', 'Microphone access was unavailable. You can upload an audio file instead.'); }
  }

  if (!progress) return result ? <><div className="signed-in"><span>Unable to load session</span><button className="text-button" onClick={onLogout}>Sign out</button></div><Result result={result} /></> : <div className="loading">Checking your volunteer progress…</div>;
  if (progress.is_complete) return <AuthenticationScreen token={token} onLogout={onLogout} completed />;

  const recording = action === 'recording';
  const freeSpeechCollected = progress.free_speech_collected ?? progress.samples_collected ?? 0;
  const freeSpeechTarget = progress.free_speech_target ?? 10;
  const pangramCollected = progress.pangram_collected ?? 0;
  const pangramTarget = progress.pangram_target ?? 10;
  const promptType = freeSpeechCollected >= freeSpeechTarget ? 'pangram' : 'free_speech';
  const collected = promptType === 'pangram' ? pangramCollected : freeSpeechCollected;
  const target = promptType === 'pangram' ? pangramTarget : freeSpeechTarget;
  const remaining = Math.max(0, target - collected);
  const sampleNumber = Math.min(collected + 1, target);
  const canSubmit = !action && collected < target && !(promptType === 'pangram' && (!pangram || pangramLoading));
  return <>
    <div className="signed-in"><span><i /> Volunteer session</span><button className="text-button" onClick={onLogout}>Sign out</button></div>
    <div className="volunteer-intro"><p className="eyebrow">Voice enrollment</p><h2>{promptType === 'free_speech' ? 'Step 1: Free speech' : 'Step 2: Read pangrams'}</h2><p>{promptType === 'free_speech' ? 'Record 10 short clips speaking naturally. Pangrams unlock when this step is complete.' : 'Free speech is complete. Read each displayed pangram aloud to finish enrollment.'}</p></div>
    <ol className="enrollment-steps" aria-label="Enrollment progress">
      <li className={promptType === 'free_speech' ? 'active' : 'complete'}><span>1</span><div><strong>Free speech</strong><small>{freeSpeechCollected} of {freeSpeechTarget} recordings</small></div></li>
      <li className={promptType === 'pangram' ? 'active' : ''}><span>2</span><div><strong>Pangrams</strong><small>{pangramCollected} of {pangramTarget} recordings</small></div></li>
    </ol>
    {promptType === 'pangram' ? <div className="pangram-card" aria-live="polite">{pangramLoading ? 'Getting your pangram…' : pangram ? <><span>Read this aloud</span><p>“{pangram.text}”</p></> : 'No pangram is available right now.'}</div> : <p className="free-speech-note">Speak naturally about anything you like for a short recording.</p>}
    <div className="progress-copy"><span>Recording {sampleNumber} of {target}</span><span>{remaining} more to finish this step</span></div>
    <div className="progress-track" aria-label={`${collected} of ${target} ${promptType} samples collected`}><div style={{ width: `${Math.min(100, (collected / target) * 100)}%` }} /></div>
    <AudioInput file={file} onChange={setFile} recording={recording} onRecord={toggleRecording} />
    {recording && <p className="recording-note">Recording in progress… {promptType === 'pangram' ? 'Read the displayed sentence, then stop.' : 'Speak naturally, then stop.'}</p>}
    <button className="enroll-button" disabled={!canSubmit} onClick={enroll}>{action === 'enroll' ? 'Saving your sample…' : collected >= target ? `${promptType === 'pangram' ? 'Pangrams' : 'Free speech'} complete` : `Submit ${promptType === 'pangram' ? 'pangram' : 'free-speech'} sample ${sampleNumber}`}</button>
    <p className="note">WAV, M4A, MP3, and WebM files are supported.</p>
    <Result result={result} />
  </>;
}

export default function App() {
  const [publicScreen, setPublicScreen] = useState('study');
  const [tab, setTab] = useState('login');
  const [accessToken, setAccessToken] = useState(() => localStorage.getItem('vas_access_token') || '');
  const [register, setRegister] = useState({ username: '', email: '', password: '' });
  const [login, setLogin] = useState({ username: '', password: '' });
  const [action, setAction] = useState('');
  const [result, setResult] = useState(null);
  const update = (setter, state, event) => setter({ ...state, [event.target.name]: event.target.value });
  const logout = () => { localStorage.removeItem('vas_access_token'); localStorage.removeItem('vas_refresh_token'); setAccessToken(''); };

  async function submitRegister(event) {
    event.preventDefault(); setAction('register'); setResult(null);
    try { const user = await api.register(register); setLogin({ username: register.username, password: register.password }); setTab('login'); setResult({ type: 'success', message: `Welcome, ${user.username}. Sign in to begin volunteering.` }); }
    catch (error) { setResult({ type: 'error', message: error.message }); } finally { setAction(''); }
  }
  async function submitLogin(event) {
    event.preventDefault(); setAction('login'); setResult(null);
    try { const tokens = await api.login(login); localStorage.setItem('vas_access_token', tokens.access); localStorage.setItem('vas_refresh_token', tokens.refresh); setAccessToken(tokens.access); }
    catch (error) { setResult({ type: 'error', message: error.message }); } finally { setAction(''); }
  }

  if (!accessToken && publicScreen === 'study') {
    return <StudyScreen onJoin={() => { setTab('register'); setPublicScreen('account'); }} onSignIn={() => { setTab('login'); setPublicScreen('account'); }} />;
  }

  return <main className="shell"><section className="hero"><p className="eyebrow">Voice Access System</p><h1>Every voice can help.</h1><p>Join our volunteer project by contributing a few short voice samples.</p></section><section className="panel">
    {accessToken ? <VolunteerFlow token={accessToken} onLogout={logout} /> : <>
      <button className="study-link" type="button" onClick={() => setPublicScreen('study')}>← About the study</button>
      <div className="tabs"><button className={tab === 'login' ? 'active' : ''} onClick={() => setTab('login')}>Sign in</button><button className={tab === 'register' ? 'active' : ''} onClick={() => setTab('register')}>Volunteer signup</button></div>
      {tab === 'login' ? <form onSubmit={submitLogin}><Field label="Username" name="username" value={login.username} onChange={(e) => update(setLogin, login, e)} /><Field label="Password" name="password" type="password" value={login.password} onChange={(e) => update(setLogin, login, e)} /><button disabled={!!action}>{action === 'login' ? 'Signing in…' : 'Continue'}</button></form> : <form onSubmit={submitRegister}><Field label="Username" name="username" value={register.username} onChange={(e) => update(setRegister, register, e)} /><Field label="Email" name="email" type="email" value={register.email} onChange={(e) => update(setRegister, register, e)} /><Field label="Password" name="password" type="password" minLength="8" value={register.password} onChange={(e) => update(setRegister, register, e)} hint="At least 8 characters" /><button disabled={!!action}>{action === 'register' ? 'Creating your account…' : 'Join as a volunteer'}</button></form>}
      <Result result={result} />
    </>}
  </section></main>;
}

function Field({ label, hint, ...props }) { return <label className="field"><span>{label}</span><input required {...props} />{hint && <small>{hint}</small>}</label>; }
