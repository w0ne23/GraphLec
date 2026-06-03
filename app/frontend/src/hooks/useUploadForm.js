import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { PHASES } from '../components/verifier/verifierConstants'
import { uploadLecture } from '../lib/api'

const DEFAULT_CATEGORY = 'etc'

function fileTitle(file) {
  return file?.name ? file.name.replace(/\.[^.]+$/, '') : ''
}

export function useUploadForm() {
  const navigate = useNavigate()

  const [file, setFile] = useState(null)
  const [title, setTitle] = useState('')
  const [mode, setMode] = useState('')
  const [phase, setPhase] = useState(PHASES.VERIFY_CHOICE)
  const [errorMessage, setErrorMessage] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  function selectFile(nextFile) {
    if (!nextFile) return
    setFile(nextFile)
    setTitle(prev => prev.trim() ? prev : fileTitle(nextFile))
  }

  function startVerify() {
    setMode('verify')
    setErrorMessage('')
    setPhase(PHASES.UPLOAD)
  }

  function startPublish() {
    setMode('publish')
    setErrorMessage('')
    setPhase(PHASES.UPLOAD)
  }

  function backToChoice() {
    setMode('')
    setErrorMessage('')
    setPhase(PHASES.VERIFY_CHOICE)
  }

  async function submit() {
    if (!file || !mode || isSubmitting) return

    const uploadTitle = title.trim() || fileTitle(file)
    setErrorMessage('')
    setIsSubmitting(true)

    try {
      const created = await uploadLecture({
        title: uploadTitle,
        category: DEFAULT_CATEGORY,
        description: '',
        file,
        workflowMode: mode,
      })

      navigate(mode === 'verify' ? `/verify/${created.id}` : `/publish/${created.id}`)
    } catch (error) {
      setIsSubmitting(false)
      setErrorMessage(String(error?.message || error))
      setPhase(PHASES.ERROR)
    }
  }

  function retry() {
    setErrorMessage('')
    setPhase(mode ? PHASES.UPLOAD : PHASES.VERIFY_CHOICE)
  }

  function reset() {
    setFile(null)
    setTitle('')
    setMode('')
    setPhase(PHASES.VERIFY_CHOICE)
    setErrorMessage('')
    setIsSubmitting(false)
  }

  return {
    file,
    title,
    mode,
    phase,
    errorMessage,
    isSubmitting,
    actions: {
      selectFile,
      setTitle,
      startVerify,
      startPublish,
      backToChoice,
      submit,
      retry,
      reset,
    },
  }
}
