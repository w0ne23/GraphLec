import raspberryLogo from '../../assets/raspberry-logo.svg'

export default function RaspberryIcon({ className = '' }) {
  return <img className={className} src={raspberryLogo} alt="" aria-hidden="true" />
}