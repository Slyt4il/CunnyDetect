<script>
  import Dropzone from "svelte-file-dropzone/Dropzone.svelte";
  import * as tf from '@tensorflow/tfjs';
  import { preprocessImage } from "./preprocess.js";
  import { predict, interpret } from "./prediction.js";

  let loadedModel;
  let status;
  let ready = false;
  let uploaded = false;
  let imgsrc;
  let finished = false;

  const onProgress = (progress) => {
    status.textContent = `Loading model... ${(progress * 100).toFixed(2)}%`;
};

tf.loadGraphModel("src/assets/js_model/model.json", { onProgress })
  .then(model => {
    status.textContent = 'Model is ready.';
    loadedModel = model;
    ready = true;
  })
  .catch(error => {
    console.error('Error loading model:', error);
  });

  let files = {
    accepted: [],
    rejected: []
  };

  async function handleFilesSelect(e) {
    const { acceptedFiles, fileRejections } = e.detail;
    files.accepted = [...files.accepted, ...acceptedFiles];
    files.rejected = [...files.rejected, ...fileRejections];

    ready = false;
    uploaded = true;

    if (files.rejected[0]) {
      status.textContent = "Invalid file(s)!"
      finished = true;
      return 0
    }

    imgsrc = URL.createObjectURL(files.accepted[0]);

    await pred();
  }

  async function pred() {
    status.textContent = "Predicting..."
    const tensor = await preprocessImage(imgsrc);
    const result = await predict(tensor, loadedModel);

    status.textContent = await interpret(result);
    finished = true;
  }

  function reset() {
    files = {
      accepted: [],
      rejected: []
    };
    ready = true;
    uploaded = false;
    finished = false;
    imgsrc = "";
    status.textContent = "Model is ready."
  }

</script>

<main>
  <h1>CunnyDetect 😭</h1>

  {#if ready}
    <div class="card">
      <Dropzone on:drop={handleFilesSelect} accept={".png,.jpg,.jpeg"} inputElement={false} multiple={false}>
        <p>Drop your waifu and let AI decide</p>
      </Dropzone>
    </div>
  {/if}
  {#if uploaded}
    <div class="card">
      <img src={imgsrc} width="224" height="224" alt="uploaded"/>
    </div>
  {/if}

    <h2 bind:this={status}>Loading model...</h2>
    {#if finished}
    <button on:click={reset}>Try Again</button>
    {/if}
  
  

  <p class="pr">
    <a href="https://github.com/Slyt4il/CunnyDetect" target="_blank" rel="noreferrer">CunnyDetect</a> works best on anime images with a single subject.
  </p>
</main>

<style>
  .pr {
    color: #888;
  }
  
  :global(body) {
    background-color: rgb(212, 252, 252);
  }

</style>
