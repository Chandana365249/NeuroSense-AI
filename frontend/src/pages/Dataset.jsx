import { useCallback, useEffect, useState } from 'react';
import { api } from '../api';
import Card from '../components/Card';
import LoadingBlock from '../components/LoadingBlock';
import Notice from '../components/Notice';
import PageHeader from '../components/PageHeader';
import Stat from '../components/Stat';
import StatusPill from '../components/StatusPill';
import { NA } from '../components/format';
import { href } from '../router';

// The public summary is aggregate only: the API omits the dataset location, file names, sizes and rating statistics.
function Summary() {
  const [state, setState] = useState({ status: 'loading' });
  const load = useCallback(async () => {
    setState({ status: 'loading' });
    try {
      setState({ status: 'ready', data: await api.datasetSummary() });
    } catch (err) {
      setState({ status: 'error', unreachable: err.status === 0, unavailable: err.status === 503 });
    }
  }, []);
  useEffect(() => { load(); }, [load]);

  if (state.status === 'loading') return <LoadingBlock label="Reading the dataset summary (this can take a few seconds the first time)…" />;
  if (state.status === 'error') {
    const retry = <button type="button" className="btn btn-secondary btn-small" onClick={load}>Retry</button>;
    if (state.unavailable) {
      return <Notice tone="warning" title="Dataset not available on this server" action={retry}>The server could not find readable DEAP participant files in its configured dataset location.</Notice>;
    }
    return <Notice tone="error" title="Could not load the dataset summary" action={retry}>{state.unreachable ? 'The service is unreachable.' : 'The summary could not be read.'}</Notice>;
  }

  const d = state.data;
  const files = d.participants || [];
  const readable = files.filter((f) => f.readable);
  const samples = readable[0]?.samples;
  const sameShape = readable.every((f) => f.samples === samples && f.trials === readable[0]?.trials);
  const seconds = typeof samples === 'number' && d.sampling_rate_hz ? samples / d.sampling_rate_hz : null;
  const authorized = d.authorized_participants || [];
  const readableIds = new Set(readable.map((f) => f.participant_id));
  const missing = authorized.filter((id) => !readableIds.has(id));
  const unexpected = [...readableIds].filter((id) => !authorized.includes(id));

  return (
    <>
      <div className="dataset-status">
        <StatusPill tone={d.participants_valid > 0 ? 'ok' : 'error'}>{d.participants_valid > 0 ? 'Available' : 'Not readable'}</StatusPill>
        <span className="muted">{d.participants_valid} of {d.participants_found} participant files readable</span>
      </div>
      <p className="muted small">
        Authorized development participants ({authorized.length}): {authorized.join(', ') || NA}
      </p>
      {missing.length || unexpected.length ? (
        <Notice tone="warning" title="Files do not match the authorized participants">
          {missing.length ? <>Not readable on this server: {missing.join(', ')}. </> : null}
          {unexpected.length ? <>Present but not authorized for development: {unexpected.join(', ')}. </> : null}
          Check the server&apos;s dataset configuration before relying on results.
        </Notice>
      ) : null}
      <div className="stats-grid stats-grid-wide">
        <Stat label="Readable participant files" value={d.participants_valid} hint={`${d.participants_found} found; not the same as a model's training participants`} />
        <Stat label="Trials" value={d.total_trials} hint={readable[0]?.trials ? `${readable[0].trials} per participant` : null} />
        <Stat label="EEG channels used" value={d.eeg_channels_used} hint={readable[0]?.channels ? `of ${readable[0].channels} channels per file` : null} />
        <Stat label="Sampling rate" value={d.sampling_rate_hz ? `${d.sampling_rate_hz} Hz` : NA} />
        <Stat label="Samples per trial" value={samples ?? NA} hint={seconds ? `${seconds} s per trial` : null} />
        <Stat label="Pre-stimulus baseline" value={typeof d.baseline_seconds === 'number' ? `${d.baseline_seconds} s` : NA} hint="removed before analysis" />
      </div>
      {!sameShape ? <p className="form-error">Participant files differ in shape; see the table below.</p> : null}

      <h3 className="subhead">Participants</h3>
      <div className="table-wrap">
        <table className="table">
          <caption className="visually-hidden">Participant files in the working dataset and their structure</caption>
          <thead>
            <tr>
              <th scope="col">Participant</th><th scope="col">Status</th><th scope="col">Trials</th>
              <th scope="col">Channels in file</th><th scope="col">Samples per channel</th>
            </tr>
          </thead>
          <tbody>
            {files.map((f) => (
              <tr key={f.participant_id}>
                <th scope="row">{f.participant_id}</th>
                <td>{f.readable ? 'Readable' : 'Could not be read'}</td>
                <td>{f.trials ?? NA}</td>
                <td>{f.channels ?? NA}</td>
                <td>{f.samples ?? NA}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {d.eeg_channel_names?.length ? (
        <>
          <h3 className="subhead">EEG channels used, in order</h3>
          <p className="channel-list">{d.eeg_channel_names.join(' · ')}</p>
        </>
      ) : null}
    </>
  );
}

export default function Dataset() {
  return (
    <>
      <PageHeader title="Dataset" subtitle="The EEG data this server works with." />
      <div className="grid-2">
        <Card title="Working dataset" titleId="working-dataset">
          <p>
            <strong>The working, trusted dataset is a documented subset of DEAP participants</strong>, listed under
            “Structure of the files” below. All models and exploratory results in this application come from them only.
            Which of them a particular model was trained on is shown, where recorded, on{' '}
            <a href={href('/model')}>Model &amp; Evidence</a> and in the <a href={href('/models')}>Model registry</a>.
          </p>
          <p className="muted">
            DEAP recorded 32 participants watching 40 one-minute music videos each. Every participant rated each video
            for valence on a 1–9 scale. The preprocessed release, used here, is downsampled to 128 Hz.
          </p>
        </Card>
        <Card title="Full dataset: decision pending" titleId="dataset-decision">
          <p>
            The source of the full DEAP dataset, and the authorization to use it, have <strong>not been decided</strong>.
            DEAP is distributed under an end-user licence agreement, and the team has not yet confirmed who will sign it
            and obtain the data.
          </p>
          <p className="muted">
            Until that decision is made, no other DEAP copy or third-party dataset is integrated. This page only reads
            the existing files and never changes them.
          </p>
        </Card>
      </div>
      <Card title="Structure of the files on this server" titleId="dataset-structure"><Summary /></Card>
    </>
  );
}
